// LoRAInjectedLinear.swift - Drop-in replacement for Linear with LoRA adaptation
// Copyright 2025 Vincent Gourbin
//
// Based on mlx-examples LoRA implementation:
// https://github.com/ml-explore/mlx-examples/tree/main/lora

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A Linear layer with Low-Rank Adaptation (LoRA)
///
/// This is a drop-in replacement for Linear that adds trainable LoRA weights.
/// During forward pass: output = Linear(x) + scale * (x @ loraA @ loraB)
///
/// The base Linear weights are frozen, only loraA and loraB are trained.
public class LoRAInjectedLinear: Linear {

    /// LoRA A matrix (down projection): [input_dim, rank] - mlx-examples convention
    /// Note: Named loraA (not lora_a) because MLXNN filters out underscore-prefixed params
    public var loraA: MLXArray

    /// LoRA B matrix (up projection): [rank, output_dim] - mlx-examples convention
    public var loraB: MLXArray

    /// LoRA rank
    public let rank: Int

    /// LoRA scale factor
    public let loraScale: Float

    /// Create a LoRAInjectedLinear from an existing Linear layer
    /// - Parameters:
    ///   - linear: The base Linear layer to wrap
    ///   - rank: LoRA rank (default 8)
    ///   - alpha: LoRA alpha for scaling (default equals rank)
    public static func fromLinear(_ linear: Linear, rank: Int = 8, alpha: Float? = nil) -> LoRAInjectedLinear {
        let outputDim = linear.weight.shape[0]
        let inputDim = linear.weight.shape[1]
        let scale = (alpha ?? Float(rank)) / Float(rank)

        return LoRAInjectedLinear(
            weight: linear.weight,
            bias: linear.bias,
            inputDim: inputDim,
            outputDim: outputDim,
            rank: rank,
            scale: scale
        )
    }

    /// Initialize LoRAInjectedLinear with existing weights
    private init(
        weight: MLXArray,
        bias: MLXArray?,
        inputDim: Int,
        outputDim: Int,
        rank: Int,
        scale: Float
    ) {
        self.rank = rank
        self.loraScale = scale

        // Initialize LoRA matrices using mlx-examples convention:
        // A: [input_dim, rank] - small random values
        // B: [rank, output_dim] - zeros
        // This way forward is: x @ loraA @ loraB (no transposes needed)
        let loraAInit = MLXRandom.uniform(
            low: -1.0 / sqrt(Float(inputDim)),
            high: 1.0 / sqrt(Float(inputDim)),
            [inputDim, rank]
        ).asType(weight.dtype)

        let loraBInit = MLXArray.zeros([rank, outputDim]).asType(weight.dtype)

        self.loraA = loraAInit
        self.loraB = loraBInit

        // Initialize base Linear with existing weights
        super.init(weight: weight, bias: bias)
    }

    /// Forward pass: base + LoRA
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear output
        let baseOutput = super.callAsFunction(x)

        // LoRA output: x @ loraA @ loraB (mlx-examples convention, no transposes)
        // loraA: [input_dim, rank], loraB: [rank, output_dim]
        let xFloat = x.asType(loraA.dtype)
        let loraOutput = matmul(matmul(xFloat, loraA), loraB)

        // Combine: base + scale * lora
        return baseOutput + loraScale * loraOutput.asType(baseOutput.dtype)
    }

    /// Get only the LoRA parameters for saving
    public var loraParameters: [String: MLXArray] {
        ["lora_a": loraA, "lora_b": loraB]
    }
}

// MARK: - Transformer LoRA Extension

extension Flux2Transformer2DModel {

    /// Convert Linear layers to LoRAInjectedLinear for training
    /// - Parameters:
    ///   - rank: LoRA rank
    ///   - alpha: LoRA alpha (scaling factor)
    ///   - targetBlocks: Which blocks to apply LoRA to (nil = all)
    public func applyLoRA(rank: Int = 8, alpha: Float? = nil, targetBlocks: LoRATargetBlocks = .all) {
        // NOTE: We do NOT use freeze() because it prevents trainableParameters() from seeing
        // the LoRA parameters even after unfreezing them. Instead, we'll filter gradients
        // in the training loop to only apply to LoRA parameters.
        //
        // IMPORTANT: We use update(modules:) instead of direct property assignment because
        // MLXNN caches module items. Direct assignment doesn't update the cache, so
        // trainableParameters() wouldn't find the new LoRA parameters.

        // Double-stream blocks
        for (idx, block) in transformerBlocks.enumerated() {
            guard targetBlocks.includesDoubleBlock(idx) else { continue }

            // Attention projections - use update(modules:) to properly update cache
            let attn = block.attn
            var modulesDict: [String: NestedItem<String, Module>] = [
                "toQ": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toQ, rank: rank, alpha: alpha)),
                "toK": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toK, rank: rank, alpha: alpha)),
                "toV": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toV, rank: rank, alpha: alpha)),
                "addQProj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.addQProj, rank: rank, alpha: alpha)),
                "addKProj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.addKProj, rank: rank, alpha: alpha)),
                "addVProj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.addVProj, rank: rank, alpha: alpha))
            ]

            if targetBlocks.includesOutputProjections {
                modulesDict["toOut"] = NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toOut, rank: rank, alpha: alpha))
                modulesDict["toAddOut"] = NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toAddOut, rank: rank, alpha: alpha))
            }

            let modules = ModuleChildren(values: modulesDict)
            attn.update(modules: modules)
        }

        // Single-stream blocks
        for (idx, block) in singleTransformerBlocks.enumerated() {
            guard targetBlocks.includesSingleBlock(idx) else { continue }

            let attn = block.attn
            var modulesDict: [String: NestedItem<String, Module>] = [
                "toQkvMlp": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toQkvMlp, rank: rank, alpha: alpha))
            ]

            if targetBlocks.includesOutputProjections {
                modulesDict["toOut"] = NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toOut, rank: rank, alpha: alpha))
            }

            let modules = ModuleChildren(values: modulesDict)
            attn.update(modules: modules)
        }

        // === NEW: Embedding and modulation layers (like fal.ai) ===

        // img_in: xEmbedder
        let xEmbedderLora = LoRAInjectedLinear.fromLinear(xEmbedder, rank: rank, alpha: alpha)
        update(modules: ModuleChildren(values: [
            "xEmbedder": NestedItem<String, Module>.value(xEmbedderLora)
        ]))

        // txt_in: contextEmbedder
        let contextEmbedderLora = LoRAInjectedLinear.fromLinear(contextEmbedder, rank: rank, alpha: alpha)
        update(modules: ModuleChildren(values: [
            "contextEmbedder": NestedItem<String, Module>.value(contextEmbedderLora)
        ]))

        // final_layer: projOut
        let projOutLora = LoRAInjectedLinear.fromLinear(projOut, rank: rank, alpha: alpha)
        update(modules: ModuleChildren(values: [
            "projOut": NestedItem<String, Module>.value(projOutLora)
        ]))

        // time_in: timeGuidanceEmbed.timestepEmbedder.linear1 and linear2
        let timestepLinear1Lora = LoRAInjectedLinear.fromLinear(timeGuidanceEmbed.timestepEmbedder.linear1, rank: rank, alpha: alpha)
        let timestepLinear2Lora = LoRAInjectedLinear.fromLinear(timeGuidanceEmbed.timestepEmbedder.linear2, rank: rank, alpha: alpha)
        timeGuidanceEmbed.timestepEmbedder.update(modules: ModuleChildren(values: [
            "linear1": NestedItem<String, Module>.value(timestepLinear1Lora),
            "linear2": NestedItem<String, Module>.value(timestepLinear2Lora)
        ]))

        // Modulation layers
        let doubleModImgLora = LoRAInjectedLinear.fromLinear(doubleStreamModulationImg.linear, rank: rank, alpha: alpha)
        doubleStreamModulationImg.update(modules: ModuleChildren(values: [
            "linear": NestedItem<String, Module>.value(doubleModImgLora)
        ]))

        let doubleModTxtLora = LoRAInjectedLinear.fromLinear(doubleStreamModulationTxt.linear, rank: rank, alpha: alpha)
        doubleStreamModulationTxt.update(modules: ModuleChildren(values: [
            "linear": NestedItem<String, Module>.value(doubleModTxtLora)
        ]))

        let singleModLora = LoRAInjectedLinear.fromLinear(singleStreamModulation.linear, rank: rank, alpha: alpha)
        singleStreamModulation.update(modules: ModuleChildren(values: [
            "linear": NestedItem<String, Module>.value(singleModLora)
        ]))

        // Log trainable parameter count
        let trainableParams = trainableParameters()
        let loraPathCount = trainableParams.flattened().filter { $0.0.hasSuffix("loraA") || $0.0.hasSuffix("loraB") }.count
        let trainableCount = trainableParams.flattenedValues().reduce(0) { $0 + $1.size }
        Flux2Debug.log("[Transformer] Applied LoRA with rank \(rank): \(loraPathCount) LoRA layers, \(trainableCount) trainable parameters")
    }

    /// Get all LoRA parameters for saving
    /// Note: Transposes weights to match inference format:
    /// - Training uses: loraA [input_dim, rank], loraB [rank, output_dim]
    /// - Inference expects: loraA [rank, input_dim], loraB [output_dim, rank]
    public func getLoRAParameters() -> [String: MLXArray] {
        var params: [String: MLXArray] = [:]

        // Double-stream blocks
        for (idx, block) in transformerBlocks.enumerated() {
            let prefix = "transformer_blocks.\(idx).attn"

            if let lora = block.attn.toQ as? LoRAInjectedLinear {
                params["\(prefix).to_q.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_q.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toK as? LoRAInjectedLinear {
                params["\(prefix).to_k.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_k.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toV as? LoRAInjectedLinear {
                params["\(prefix).to_v.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_v.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear {
                params["\(prefix).add_q_proj.lora_A.weight"] = lora.loraA.T
                params["\(prefix).add_q_proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear {
                params["\(prefix).add_k_proj.lora_A.weight"] = lora.loraA.T
                params["\(prefix).add_k_proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear {
                params["\(prefix).add_v_proj.lora_A.weight"] = lora.loraA.T
                params["\(prefix).add_v_proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                params["\(prefix).to_out.0.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_out.0.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear {
                params["\(prefix).to_add_out.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_add_out.lora_B.weight"] = lora.loraB.T
            }
        }

        // Single-stream blocks
        for (idx, block) in singleTransformerBlocks.enumerated() {
            let prefix = "single_transformer_blocks.\(idx).attn"

            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear {
                params["\(prefix).to_qkv_mlp.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_qkv_mlp.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                params["\(prefix).to_out.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_out.lora_B.weight"] = lora.loraB.T
            }
        }

        // === NEW: Embedding and modulation layers ===

        // img_in: xEmbedder
        if let lora = xEmbedder as? LoRAInjectedLinear {
            params["img_in.lora_A.weight"] = lora.loraA.T
            params["img_in.lora_B.weight"] = lora.loraB.T
        }

        // txt_in: contextEmbedder
        if let lora = contextEmbedder as? LoRAInjectedLinear {
            params["txt_in.lora_A.weight"] = lora.loraA.T
            params["txt_in.lora_B.weight"] = lora.loraB.T
        }

        // final_layer: projOut
        if let lora = projOut as? LoRAInjectedLinear {
            params["final_layer.linear.lora_A.weight"] = lora.loraA.T
            params["final_layer.linear.lora_B.weight"] = lora.loraB.T
        }

        // time_in: timestepEmbedder linear1 and linear2
        if let lora = timeGuidanceEmbed.timestepEmbedder.linear1 as? LoRAInjectedLinear {
            params["time_in.in_layer.lora_A.weight"] = lora.loraA.T
            params["time_in.in_layer.lora_B.weight"] = lora.loraB.T
        }
        if let lora = timeGuidanceEmbed.timestepEmbedder.linear2 as? LoRAInjectedLinear {
            params["time_in.out_layer.lora_A.weight"] = lora.loraA.T
            params["time_in.out_layer.lora_B.weight"] = lora.loraB.T
        }

        // Modulation layers
        if let lora = doubleStreamModulationImg.linear as? LoRAInjectedLinear {
            params["double_stream_modulation_img.lin.lora_A.weight"] = lora.loraA.T
            params["double_stream_modulation_img.lin.lora_B.weight"] = lora.loraB.T
        }
        if let lora = doubleStreamModulationTxt.linear as? LoRAInjectedLinear {
            params["double_stream_modulation_txt.lin.lora_A.weight"] = lora.loraA.T
            params["double_stream_modulation_txt.lin.lora_B.weight"] = lora.loraB.T
        }
        if let lora = singleStreamModulation.linear as? LoRAInjectedLinear {
            params["single_stream_modulation.lin.lora_A.weight"] = lora.loraA.T
            params["single_stream_modulation.lin.lora_B.weight"] = lora.loraB.T
        }

        return params
    }

    /// Count trainable LoRA parameters
    public var loraParameterCount: Int {
        var count = 0
        for (_, array) in getLoRAParameters() {
            count += array.size
        }
        return count
    }

    /// Unfreeze only LoRA parameters (loraA and loraB)
    /// Call this after freeze(recursive: true) to make only LoRA params trainable
    public func unfreezeLoRAParameters() {
        // Unfreeze LoRA in double-stream blocks
        for block in transformerBlocks {
            if let lora = block.attn.toQ as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toK as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toV as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
        }

        // Unfreeze LoRA in single-stream blocks
        for block in singleTransformerBlocks {
            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
        }

        // Unfreeze LoRA in embedding and modulation layers
        if let lora = xEmbedder as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = contextEmbedder as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = projOut as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = timeGuidanceEmbed.timestepEmbedder.linear1 as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = timeGuidanceEmbed.timestepEmbedder.linear2 as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = doubleStreamModulationImg.linear as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = doubleStreamModulationTxt.linear as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }
        if let lora = singleStreamModulation.linear as? LoRAInjectedLinear {
            lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
        }

        Flux2Debug.log("[Transformer] Unfroze LoRA parameters (loraA, loraB)")
    }
}

// MARK: - Target Blocks Configuration

/// Configuration for which blocks to apply LoRA to
public struct LoRATargetBlocks: Sendable {
    public let doubleBlockIndices: [Int]?  // nil = all
    public let singleBlockIndices: [Int]?  // nil = all
    public let includesOutputProjections: Bool
    public let includesFFN: Bool

    public init(
        doubleBlockIndices: [Int]? = nil,
        singleBlockIndices: [Int]? = nil,
        includesOutputProjections: Bool = false,
        includesFFN: Bool = false
    ) {
        self.doubleBlockIndices = doubleBlockIndices
        self.singleBlockIndices = singleBlockIndices
        self.includesOutputProjections = includesOutputProjections
        self.includesFFN = includesFFN
    }

    public static let all = LoRATargetBlocks(
        doubleBlockIndices: nil,
        singleBlockIndices: nil,
        includesOutputProjections: true,
        includesFFN: true
    )

    public static let attentionOnly = LoRATargetBlocks(
        doubleBlockIndices: nil,
        singleBlockIndices: nil,
        includesOutputProjections: false,
        includesFFN: false
    )

    public static let attentionWithOutput = LoRATargetBlocks(
        doubleBlockIndices: nil,
        singleBlockIndices: nil,
        includesOutputProjections: true,
        includesFFN: false
    )

    public func includesDoubleBlock(_ index: Int) -> Bool {
        doubleBlockIndices == nil || doubleBlockIndices!.contains(index)
    }

    public func includesSingleBlock(_ index: Int) -> Bool {
        singleBlockIndices == nil || singleBlockIndices!.contains(index)
    }
}

// MARK: - LoRATargetLayers to LoRATargetBlocks Conversion

extension LoRATargetLayers {
    /// Convert to LoRATargetBlocks for use in applyLoRA
    public func toTargetBlocks() -> LoRATargetBlocks {
        switch self {
        case .attention:
            return .attentionOnly
        case .attentionOutput:
            return .attentionWithOutput
        case .attentionFFN:
            return LoRATargetBlocks(
                includesOutputProjections: true,
                includesFFN: true
            )
        case .all:
            return .all
        }
    }
}
