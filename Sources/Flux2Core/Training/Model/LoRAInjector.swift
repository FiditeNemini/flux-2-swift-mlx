// LoRAInjector.swift - Inject trainable LoRA layers into transformer
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Manages injection of trainable LoRA layers into a Flux2 transformer
///
/// The LoRAInjector creates trainable LoRA layers that wrap the base model's
/// Linear layers. These LoRA layers are tracked separately and their outputs
/// are applied additively during forward pass.
///
/// Usage:
/// 1. Create injector with config
/// 2. Call `injectLoRA(into:)` to analyze transformer and create LoRA layers
/// 3. During training, use `applyLoRA(baseOutput:input:layerPath:)` to get adapted outputs
/// 4. Call `saveWeights(to:)` to save trained LoRA
public class LoRAInjector: @unchecked Sendable {
    
    /// Injected LoRA layers (path -> layer)
    private(set) var loraLayers: [String: TrainableLoRALinear] = [:]
    
    /// Configuration
    public let config: LoRATrainingConfig
    
    /// Target model type
    public let modelType: Flux2Model
    
    /// Initialize LoRA injector
    /// - Parameters:
    ///   - config: Training configuration
    ///   - modelType: Target model type
    public init(config: LoRATrainingConfig, modelType: Flux2Model) {
        self.config = config
        self.modelType = modelType
    }
    
    // MARK: - Layer Injection
    
    /// Inject LoRA layers into transformer
    ///
    /// This method analyzes the transformer structure and creates TrainableLoRALinear
    /// layers for each target layer. The LoRA layers wrap copies of the base weights
    /// and are stored in this injector for use during training.
    ///
    /// - Parameter transformer: The transformer model to inject into
    /// - Returns: List of injected layer paths
    @discardableResult
    public func injectLoRA(into transformer: Flux2Transformer2DModel) -> [String] {
        var injectedPaths: [String] = []
        let transformerConfig = modelType.transformerConfig
        
        // Determine which layers to target based on config
        let targetLayers = getTargetLayerPaths(config: transformerConfig)
        
        Flux2Debug.log("[LoRAInjector] Injecting LoRA into \(targetLayers.count) layers")
        
        // Inject into double-stream blocks
        for blockIdx in 0..<transformerConfig.numLayers {
            let block = transformer.transformerBlocks[blockIdx]
            
            // Attention layers
            if shouldInject("transformerBlocks.\(blockIdx).attn.toQ") {
                createLoRAForLinear(
                    block.attn.toQ,
                    path: "transformerBlocks.\(blockIdx).attn.toQ"
                )
                injectedPaths.append("transformerBlocks.\(blockIdx).attn.toQ")
            }
            
            if shouldInject("transformerBlocks.\(blockIdx).attn.toK") {
                createLoRAForLinear(
                    block.attn.toK,
                    path: "transformerBlocks.\(blockIdx).attn.toK"
                )
                injectedPaths.append("transformerBlocks.\(blockIdx).attn.toK")
            }
            
            if shouldInject("transformerBlocks.\(blockIdx).attn.toV") {
                createLoRAForLinear(
                    block.attn.toV,
                    path: "transformerBlocks.\(blockIdx).attn.toV"
                )
                injectedPaths.append("transformerBlocks.\(blockIdx).attn.toV")
            }
            
            // Text/context attention
            if shouldInject("transformerBlocks.\(blockIdx).attn.addQProj") {
                createLoRAForLinear(
                    block.attn.addQProj,
                    path: "transformerBlocks.\(blockIdx).attn.addQProj"
                )
                injectedPaths.append("transformerBlocks.\(blockIdx).attn.addQProj")
            }
            
            if shouldInject("transformerBlocks.\(blockIdx).attn.addKProj") {
                createLoRAForLinear(
                    block.attn.addKProj,
                    path: "transformerBlocks.\(blockIdx).attn.addKProj"
                )
                injectedPaths.append("transformerBlocks.\(blockIdx).attn.addKProj")
            }
            
            if shouldInject("transformerBlocks.\(blockIdx).attn.addVProj") {
                createLoRAForLinear(
                    block.attn.addVProj,
                    path: "transformerBlocks.\(blockIdx).attn.addVProj"
                )
                injectedPaths.append("transformerBlocks.\(blockIdx).attn.addVProj")
            }
            
            // Output projections
            if config.targetLayers.includesOutputProjections {
                if shouldInject("transformerBlocks.\(blockIdx).attn.toOut") {
                    createLoRAForLinear(
                        block.attn.toOut,
                        path: "transformerBlocks.\(blockIdx).attn.toOut"
                    )
                    injectedPaths.append("transformerBlocks.\(blockIdx).attn.toOut")
                }
                
                if shouldInject("transformerBlocks.\(blockIdx).attn.toAddOut") {
                    createLoRAForLinear(
                        block.attn.toAddOut,
                        path: "transformerBlocks.\(blockIdx).attn.toAddOut"
                    )
                    injectedPaths.append("transformerBlocks.\(blockIdx).attn.toAddOut")
                }
            }
        }
        
        // Inject into single-stream blocks
        for blockIdx in 0..<transformerConfig.numSingleLayers {
            let block = transformer.singleTransformerBlocks[blockIdx]
            
            // Single block combined QKV+MLP projection
            if shouldInject("singleTransformerBlocks.\(blockIdx).attn.toQkvMlp") {
                createLoRAForLinear(
                    block.attn.toQkvMlp,
                    path: "singleTransformerBlocks.\(blockIdx).attn.toQkvMlp"
                )
                injectedPaths.append("singleTransformerBlocks.\(blockIdx).attn.toQkvMlp")
            }
            
            if config.targetLayers.includesOutputProjections {
                if shouldInject("singleTransformerBlocks.\(blockIdx).attn.toOut") {
                    createLoRAForLinear(
                        block.attn.toOut,
                        path: "singleTransformerBlocks.\(blockIdx).attn.toOut"
                    )
                    injectedPaths.append("singleTransformerBlocks.\(blockIdx).attn.toOut")
                }
            }
        }
        
        Flux2Debug.log("[LoRAInjector] Injected \(injectedPaths.count) LoRA layers")
        return injectedPaths
    }
    
    /// Create a LoRA layer for a given Linear layer
    /// - Parameters:
    ///   - linear: The base Linear layer to wrap
    ///   - path: The layer path identifier
    private func createLoRAForLinear(_ linear: Linear, path: String) {
        let lora = TrainableLoRALinear(
            wrapping: linear,
            rank: config.rank,
            alpha: config.alpha,
            dropoutRate: config.dropout
        )
        // Note: base is frozen automatically in TrainableLoRALinear.init

        loraLayers[path] = lora
    }
    
    // MARK: - Forward Pass Support
    
    /// Apply LoRA adaptation to a layer's output
    ///
    /// Call this during training to get the LoRA-adapted output for a layer.
    /// If no LoRA exists for the path, returns the base output unchanged.
    ///
    /// - Parameters:
    ///   - baseOutput: Output from the base Linear layer
    ///   - input: Input that was fed to the Linear layer
    ///   - layerPath: Path identifying which layer
    /// - Returns: Output with LoRA adaptation applied
    public func applyLoRA(baseOutput: MLXArray, input: MLXArray, layerPath: String) -> MLXArray {
        guard let lora = loraLayers[layerPath] else {
            return baseOutput
        }
        
        // Compute LoRA output: scale * (input @ loraA.T @ loraB.T)
        let loraOutput = matmul(matmul(input, lora.loraA.T), lora.loraB.T)
        
        return baseOutput + lora.scale * loraOutput
    }
    
    /// Get LoRA layer for a path (for direct forward pass)
    public func getLoRALayer(for path: String) -> TrainableLoRALinear? {
        loraLayers[path]
    }
    
    /// Check if a layer path should have LoRA injected
    private func shouldInject(_ path: String) -> Bool {
        let targets = config.targetLayers
        
        // Attention Q, K, V are always included
        if path.contains(".toQ") || path.contains(".toK") || path.contains(".toV") ||
           path.contains(".addQProj") || path.contains(".addKProj") || path.contains(".addVProj") ||
           path.contains(".toQkvMlp") {
            return true
        }
        
        // Output projections
        if targets.includesOutputProjections {
            if path.contains(".toOut") || path.contains(".toAddOut") {
                return true
            }
        }
        
        // FFN layers (not yet implemented in this version)
        if targets.includesFFN {
            // Would include ff.*, ffContext.* paths
        }
        
        return false
    }
    
    /// Get list of target layer paths based on configuration
    private func getTargetLayerPaths(config: Flux2TransformerConfig) -> [String] {
        var paths: [String] = []
        
        // Double-stream blocks
        for i in 0..<config.numLayers {
            // Image attention
            paths.append("transformerBlocks.\(i).attn.toQ")
            paths.append("transformerBlocks.\(i).attn.toK")
            paths.append("transformerBlocks.\(i).attn.toV")
            
            // Text attention
            paths.append("transformerBlocks.\(i).attn.addQProj")
            paths.append("transformerBlocks.\(i).attn.addKProj")
            paths.append("transformerBlocks.\(i).attn.addVProj")
            
            if self.config.targetLayers.includesOutputProjections {
                paths.append("transformerBlocks.\(i).attn.toOut")
                paths.append("transformerBlocks.\(i).attn.toAddOut")
            }
        }
        
        // Single-stream blocks
        for i in 0..<config.numSingleLayers {
            paths.append("singleTransformerBlocks.\(i).attn.toQkvMlp")
            
            if self.config.targetLayers.includesOutputProjections {
                paths.append("singleTransformerBlocks.\(i).attn.toOut")
            }
        }
        
        return paths
    }
    
    // MARK: - Parameter Access
    
    /// Get all trainable LoRA parameters
    public func trainableParameters() -> [String: MLXArray] {
        var params: [String: MLXArray] = [:]
        
        for (path, lora) in loraLayers {
            params["\(path).lora_A"] = lora.loraA
            params["\(path).lora_B"] = lora.loraB
        }
        
        return params
    }
    
    /// Get flattened parameter array for optimizer
    public func flattenedParameters() -> [MLXArray] {
        var arrays: [MLXArray] = []
        
        for (_, lora) in loraLayers.sorted(by: { $0.key < $1.key }) {
            arrays.append(lora.loraA)
            arrays.append(lora.loraB)
        }
        
        return arrays
    }
    
    /// Update parameters from flattened array
    public func updateParameters(from arrays: [MLXArray]) {
        let sortedKeys = loraLayers.keys.sorted()
        var idx = 0
        
        for key in sortedKeys {
            guard let lora = loraLayers[key] else { continue }
            lora.loraA = arrays[idx]
            lora.loraB = arrays[idx + 1]
            idx += 2
        }
    }
    
    /// Total number of trainable parameters
    public var trainableParameterCount: Int {
        loraLayers.values.reduce(0) { count, lora in
            count + lora.loraA.size + lora.loraB.size
        }
    }
    
    /// Estimated memory for trainable parameters in MB
    public var trainableParameterMemoryMB: Float {
        Float(trainableParameterCount * 4) / (1024 * 1024)  // Float32 = 4 bytes
    }
    
    // MARK: - LoRA Management

    /// Merge all LoRA weights into base model
    /// Note: This is a no-op in the current implementation since base weights
    /// are stored as immutable copies. For inference, use the LoRA adapter directly.
    public func mergeIntoBase() {
        // Not supported - base weights are stored as immutable copies
        Flux2Debug.log("[LoRAInjector] Merge not supported - use LoRA adapter for inference")
    }

    /// Enable/disable all LoRA layers
    public func setEnabled(_ enabled: Bool) {
        for (_, lora) in loraLayers {
            lora.enabled = enabled
        }
    }
    
    /// Get statistics about injected LoRA
    public func getStatistics() -> LoRAInjectionStatistics {
        let totalParams = trainableParameterCount
        let numLayers = loraLayers.count
        
        return LoRAInjectionStatistics(
            numLayers: numLayers,
            rank: config.rank,
            alpha: config.alpha,
            totalTrainableParameters: totalParams,
            memorySizeMB: trainableParameterMemoryMB
        )
    }
    
    // MARK: - Serialization
    
    /// Save LoRA weights to safetensors file
    public func saveWeights(to url: URL) throws {
        var weights: [String: MLXArray] = [:]
        
        for (path, lora) in loraLayers {
            // Convert path to standard format
            let diffusersPath = swiftPathToDiffusers(path)
            weights["\(diffusersPath).lora_A.weight"] = lora.loraA
            weights["\(diffusersPath).lora_B.weight"] = lora.loraB
        }
        
        // Add metadata
        // Note: MLX save doesn't support metadata directly, would need custom format
        
        try save(arrays: weights, url: url)
        Flux2Debug.log("[LoRAInjector] Saved weights to \(url.path)")
    }
    
    /// Load LoRA weights from safetensors file
    public func loadWeights(from url: URL) throws {
        let weights = try loadArrays(url: url)
        
        // Map loaded weights to our layers
        for (path, lora) in loraLayers {
            let diffusersPath = swiftPathToDiffusers(path)
            
            if let loraA = weights["\(diffusersPath).lora_A.weight"] {
                lora.loraA = loraA
            }
            
            if let loraB = weights["\(diffusersPath).lora_B.weight"] {
                lora.loraB = loraB
            }
        }
        
        Flux2Debug.log("[LoRAInjector] Loaded weights from \(url.path)")
    }
    
    /// Convert Swift layer path to Diffusers format
    private func swiftPathToDiffusers(_ swiftPath: String) -> String {
        var path = swiftPath
        
        // Transform block names
        path = path.replacingOccurrences(of: "transformerBlocks.", with: "transformer.transformer_blocks.")
        path = path.replacingOccurrences(of: "singleTransformerBlocks.", with: "transformer.single_transformer_blocks.")
        
        // Transform attention names
        path = path.replacingOccurrences(of: ".attn.toQ", with: ".attn.to_q")
        path = path.replacingOccurrences(of: ".attn.toK", with: ".attn.to_k")
        path = path.replacingOccurrences(of: ".attn.toV", with: ".attn.to_v")
        path = path.replacingOccurrences(of: ".attn.toOut", with: ".attn.to_out.0")
        path = path.replacingOccurrences(of: ".attn.addQProj", with: ".attn.add_q_proj")
        path = path.replacingOccurrences(of: ".attn.addKProj", with: ".attn.add_k_proj")
        path = path.replacingOccurrences(of: ".attn.addVProj", with: ".attn.add_v_proj")
        path = path.replacingOccurrences(of: ".attn.toAddOut", with: ".attn.to_add_out")
        path = path.replacingOccurrences(of: ".attn.toQkvMlp", with: ".attn.to_qkv_mlp_proj")
        
        return path
    }
}

// MARK: - Statistics

/// Statistics about injected LoRA layers
public struct LoRAInjectionStatistics: Sendable {
    public let numLayers: Int
    public let rank: Int
    public let alpha: Float
    public let totalTrainableParameters: Int
    public let memorySizeMB: Float
    
    public var summary: String {
        """
        LoRA Injection Statistics:
          Layers: \(numLayers)
          Rank: \(rank), Alpha: \(alpha), Scale: \(alpha / Float(rank))
          Trainable parameters: \(formatNumber(totalTrainableParameters))
          Memory: \(String(format: "%.1f", memorySizeMB)) MB
        """
    }
    
    private func formatNumber(_ n: Int) -> String {
        if n >= 1_000_000 {
            return String(format: "%.1fM", Float(n) / 1_000_000)
        } else if n >= 1_000 {
            return String(format: "%.1fK", Float(n) / 1_000)
        }
        return "\(n)"
    }
}
