// EMAManager.swift - Exponential Moving Average for LoRA weights
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Manages Exponential Moving Average (EMA) of LoRA weights during training
///
/// EMA smooths the weight updates over time, reducing noise and often leading
/// to better generalization. The formula is:
/// `ema_weight = decay * ema_weight + (1 - decay) * current_weight`
///
/// Typical decay values:
/// - 0.99: Fast averaging, more responsive to recent updates
/// - 0.999: Slow averaging, smoother but lags behind
/// - 0.9999: Very slow averaging, for very long training runs
public final class EMAManager: @unchecked Sendable {

    // MARK: - Properties

    /// EMA decay factor (0.99 - 0.9999)
    public let decay: Float

    /// Current EMA weights (path -> weight tensor)
    private var emaWeights: [String: MLXArray]

    /// Number of updates performed
    private var updateCount: Int = 0

    /// Whether EMA has been initialized
    public var isInitialized: Bool {
        !emaWeights.isEmpty
    }

    // MARK: - Initialization

    /// Initialize EMA manager
    /// - Parameter decay: EMA decay factor (default 0.99)
    public init(decay: Float = 0.99) {
        precondition(decay > 0 && decay < 1, "EMA decay must be between 0 and 1")
        self.decay = decay
        self.emaWeights = [:]
    }

    // MARK: - Core Operations

    /// Initialize EMA weights from transformer's current LoRA parameters
    /// Should be called once at the start of training
    /// - Parameter transformer: Transformer with injected LoRA layers
    public func initialize(from transformer: Flux2Transformer2DModel) {
        let loraParams = transformer.getLoRAParameters()

        for (path, weight) in loraParams {
            // Create a copy of the weight for EMA tracking
            emaWeights[path] = weight
        }

        updateCount = 0
        Flux2Debug.log("[EMA] Initialized with \(emaWeights.count) weight tensors (decay=\(decay))")
    }

    /// Update EMA weights with current LoRA weights from transformer
    /// Call this after each optimizer step
    /// - Parameter transformer: Transformer with updated LoRA weights
    public func update(from transformer: Flux2Transformer2DModel) {
        guard isInitialized else {
            Flux2Debug.log("[EMA] Warning: EMA not initialized, skipping update")
            return
        }

        let currentParams = transformer.getLoRAParameters()

        for (path, currentWeight) in currentParams {
            guard let emaWeight = emaWeights[path] else {
                // New parameter, initialize it
                emaWeights[path] = currentWeight
                continue
            }

            // EMA update: ema = decay * ema + (1 - decay) * current
            let oneMinusDecay = 1.0 - decay
            emaWeights[path] = decay * emaWeight + oneMinusDecay * currentWeight
        }

        updateCount += 1

        // Evaluate EMA weights periodically to prevent graph accumulation
        if updateCount % 10 == 0 {
            eval(Array(emaWeights.values))
        }
    }

    /// Get the current EMA weights
    /// - Returns: Dictionary of path -> EMA weight tensor
    public func getEMAWeights() -> [String: MLXArray] {
        // Ensure all weights are evaluated before returning
        eval(Array(emaWeights.values))
        return emaWeights
    }

    /// Apply EMA weights to the transformer (for inference or saving)
    /// - Parameter transformer: Transformer to update with EMA weights
    /// - Returns: The original weights (for restoration)
    public func applyToTransformer(_ transformer: Flux2Transformer2DModel) -> [String: MLXArray] {
        guard isInitialized else {
            Flux2Debug.log("[EMA] Warning: EMA not initialized, cannot apply")
            return [:]
        }

        // Store original weights
        let originalWeights = transformer.getLoRAParameters()

        // Apply EMA weights
        applyWeightsToTransformer(transformer, weights: emaWeights)

        return originalWeights
    }

    /// Restore original weights to transformer after using EMA weights
    /// - Parameters:
    ///   - transformer: Transformer to restore
    ///   - originalWeights: Original weights from applyToTransformer
    public func restoreTransformer(_ transformer: Flux2Transformer2DModel, with originalWeights: [String: MLXArray]) {
        applyWeightsToTransformer(transformer, weights: originalWeights)
    }

    /// Save EMA weights to safetensors file
    /// - Parameter url: Path to save weights
    public func saveWeights(to url: URL) throws {
        guard isInitialized else {
            throw EMAError.notInitialized
        }

        try save(arrays: emaWeights, url: url)
        Flux2Debug.log("[EMA] Saved EMA weights to \(url.path) (\(emaWeights.count) tensors)")
    }

    /// Load EMA weights from safetensors file
    /// - Parameter url: Path to load weights from
    public func loadWeights(from url: URL) throws {
        let loaded = try loadArrays(url: url)
        emaWeights = loaded
        Flux2Debug.log("[EMA] Loaded EMA weights from \(url.path) (\(emaWeights.count) tensors)")
    }

    // MARK: - Private Helpers

    /// Apply weights dictionary to transformer's LoRA layers
    private func applyWeightsToTransformer(_ transformer: Flux2Transformer2DModel, weights: [String: MLXArray]) {
        // Update double-stream blocks
        for (blockIdx, block) in transformer.transformerBlocks.enumerated() {
            updateLoRALayer(block.attn.toQ, basePath: "transformer_blocks.\(blockIdx).attn.to_q", weights: weights)
            updateLoRALayer(block.attn.toK, basePath: "transformer_blocks.\(blockIdx).attn.to_k", weights: weights)
            updateLoRALayer(block.attn.toV, basePath: "transformer_blocks.\(blockIdx).attn.to_v", weights: weights)
            updateLoRALayer(block.attn.addQProj, basePath: "transformer_blocks.\(blockIdx).attn.add_q_proj", weights: weights)
            updateLoRALayer(block.attn.addKProj, basePath: "transformer_blocks.\(blockIdx).attn.add_k_proj", weights: weights)
            updateLoRALayer(block.attn.addVProj, basePath: "transformer_blocks.\(blockIdx).attn.add_v_proj", weights: weights)
            updateLoRALayer(block.attn.toOut, basePath: "transformer_blocks.\(blockIdx).attn.to_out.0", weights: weights)
            updateLoRALayer(block.attn.toAddOut, basePath: "transformer_blocks.\(blockIdx).attn.to_add_out", weights: weights)
        }

        // Update single-stream blocks
        for (blockIdx, block) in transformer.singleTransformerBlocks.enumerated() {
            updateLoRALayer(block.attn.toQkvMlp, basePath: "single_transformer_blocks.\(blockIdx).attn.to_qkv_mlp", weights: weights)
            updateLoRALayer(block.attn.toOut, basePath: "single_transformer_blocks.\(blockIdx).attn.to_out.0", weights: weights)
        }
    }

    /// Update a single LoRA layer with weights from dictionary
    private func updateLoRALayer(_ layer: Linear, basePath: String, weights: [String: MLXArray]) {
        guard let lora = layer as? LoRAInjectedLinear else { return }

        let loraAPath = "\(basePath).lora_A.weight"
        let loraBPath = "\(basePath).lora_B.weight"

        if let loraA = weights[loraAPath] {
            lora.loraA = loraA
        }
        if let loraB = weights[loraBPath] {
            lora.loraB = loraB
        }
    }

    // MARK: - Statistics

    /// Get EMA statistics
    public var statistics: String {
        """
        EMA Statistics:
          Decay: \(decay)
          Weights tracked: \(emaWeights.count)
          Updates performed: \(updateCount)
        """
    }
}

// MARK: - Errors

public enum EMAError: Error, LocalizedError {
    case notInitialized
    case weightsNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "EMA manager not initialized. Call initialize() first."
        case .weightsNotFound(let path):
            return "EMA weights not found at path: \(path)"
        }
    }
}
