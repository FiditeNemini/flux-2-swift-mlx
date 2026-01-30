// TrainableLoRALinear.swift - Trainable LoRA layer for gradient computation
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A trainable LoRA linear layer for fine-tuning
///
/// This implements LoRA (Low-Rank Adaptation) as a trainable module:
/// `output = base_linear(x) + scale * dropout(x @ loraA.T @ loraB.T)`
///
/// The base linear weights are stored but NOT as a child Module to avoid
/// MLX module tree conflicts. Only loraA and loraB are tracked as trainable parameters.
public class TrainableLoRALinear: Module, UnaryLayer {

    /// Base weight matrix (frozen, stored directly - not as a Module child)
    /// Shape: [output_dim, input_dim]
    public let baseWeight: MLXArray

    /// Base bias (optional)
    public let baseBias: MLXArray?

    /// LoRA A matrix (down projection): [rank, input_dim]
    @ParameterInfo(key: "lora_A") public var loraA: MLXArray

    /// LoRA B matrix (up projection): [output_dim, rank]
    @ParameterInfo(key: "lora_B") public var loraB: MLXArray

    /// LoRA rank
    public let rank: Int

    /// LoRA alpha for scaling
    public let alpha: Float

    /// Dropout rate (0.0 to disable)
    public let dropoutRate: Float

    /// Scale factor (alpha / rank)
    public var scale: Float {
        alpha / Float(rank)
    }

    /// Whether LoRA is enabled (can be disabled during validation)
    public var enabled: Bool = true

    /// Initialize trainable LoRA layer
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - outputDim: Output dimension
    ///   - rank: LoRA rank
    ///   - alpha: LoRA alpha for scaling
    ///   - dropoutRate: Dropout rate for regularization
    ///   - baseWeight: Optional pre-existing base weights
    ///   - baseBias: Optional pre-existing base bias
    public init(
        inputDim: Int,
        outputDim: Int,
        rank: Int = 16,
        alpha: Float = 16.0,
        dropoutRate: Float = 0.0,
        baseWeight: MLXArray? = nil,
        baseBias: MLXArray? = nil
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropoutRate = dropoutRate

        // Store base weights directly (not as Module)
        if let weight = baseWeight {
            self.baseWeight = weight
        } else {
            // Initialize random weights if not provided
            let k = 1.0 / Float(inputDim)
            self.baseWeight = MLXRandom.uniform(
                low: -sqrt(k),
                high: sqrt(k),
                [outputDim, inputDim]
            ).asType(.float32)
        }
        self.baseBias = baseBias

        // Initialize LoRA matrices
        // A: Kaiming uniform initialization
        // B: Zero initialization (so LoRA starts as identity)
        let stdA = sqrt(2.0 / Float(inputDim))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -stdA,
            high: stdA,
            [rank, inputDim]
        ).asType(.float32)

        self._loraB.wrappedValue = MLXArray.zeros([outputDim, rank]).asType(.float32)

        super.init()
    }

    /// Initialize from existing linear layer (copies weights, doesn't reference the module)
    /// - Parameters:
    ///   - linear: Existing linear layer to copy weights from
    ///   - rank: LoRA rank
    ///   - alpha: LoRA alpha
    ///   - dropoutRate: Dropout rate
    public init(
        wrapping linear: Linear,
        rank: Int = 16,
        alpha: Float = 16.0,
        dropoutRate: Float = 0.0
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropoutRate = dropoutRate

        // Copy weights from original Linear (don't keep a reference to the Module)
        self.baseWeight = linear.weight
        self.baseBias = linear.bias

        // Get input dimension from weight shape
        let inputDim = linear.weight.shape[1]
        let outputDim = linear.weight.shape[0]

        // Initialize LoRA matrices
        let stdA = sqrt(2.0 / Float(inputDim))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -stdA,
            high: stdA,
            [rank, inputDim]
        ).asType(.float32)

        self._loraB.wrappedValue = MLXArray.zeros([outputDim, rank]).asType(.float32)

        super.init()
    }

    /// Forward pass with LoRA adaptation
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear output: x @ weight.T + bias
        var baseOutput = matmul(x, baseWeight.T)
        if let bias = baseBias {
            baseOutput = baseOutput + bias
        }

        // If LoRA disabled, return base only
        guard enabled else {
            return baseOutput
        }

        // LoRA computation: x @ loraA.T @ loraB.T
        var loraInput = x

        // Apply dropout during training (if rate > 0)
        if dropoutRate > 0 && TrainingMode.shared.isTraining {
            let mask = MLXRandom.uniform(low: Float(0), high: Float(1), x.shape) .> dropoutRate
            loraInput = loraInput * mask.asType(x.dtype) / (1 - dropoutRate)
        }

        // LoRA forward: scale * (x @ loraA.T @ loraB.T)
        let loraOutput = matmul(matmul(loraInput, loraA.T), loraB.T)

        return baseOutput + scale * loraOutput
    }

    /// Compute base linear output only (for comparison)
    public func base(_ x: MLXArray) -> MLXArray {
        var output = matmul(x, baseWeight.T)
        if let bias = baseBias {
            output = output + bias
        }
        return output
    }

    /// Get the LoRA delta weight (for merging or analysis)
    public var deltaWeight: MLXArray {
        // delta_W = scale * loraB @ loraA
        scale * matmul(loraB, loraA)
    }

    /// Get only the trainable LoRA parameters
    public func loraParameters() -> [String: MLXArray] {
        [
            "lora_A": loraA,
            "lora_B": loraB
        ]
    }

    /// Update LoRA parameters directly (bypasses Module update mechanism)
    /// This is used during training to avoid MLX Module system issues
    public func updateLoRA(newLoraA: MLXArray, newLoraB: MLXArray) {
        _loraA.wrappedValue = newLoraA
        _loraB.wrappedValue = newLoraB
    }
}

// MARK: - LoRA Initialization Strategies

/// Initialization strategy for LoRA matrices
public enum LoRAInitialization {
    /// Standard: A=Kaiming, B=Zero
    case standard

    /// Gaussian: A=N(0, 1/rank), B=Zero
    case gaussian

    /// Xavier: A=Xavier uniform, B=Zero
    case xavier

    /// Initialize A matrix
    func initializeA(shape: [Int], inputDim: Int) -> MLXArray {
        let rank = shape[0]

        switch self {
        case .standard:
            let std = sqrt(2.0 / Float(inputDim))
            return MLXRandom.uniform(low: -std, high: std, shape).asType(.float32)

        case .gaussian:
            let std = 1.0 / sqrt(Float(rank))
            return MLXRandom.normal(shape) * std

        case .xavier:
            let bound = sqrt(6.0 / Float(inputDim + rank))
            return MLXRandom.uniform(low: -bound, high: bound, shape).asType(.float32)
        }
    }

    /// Initialize B matrix (always zeros for identity start)
    func initializeB(shape: [Int]) -> MLXArray {
        MLXArray.zeros(shape).asType(.float32)
    }
}

// MARK: - Batch LoRA Creation

extension TrainableLoRALinear {

    /// Create multiple LoRA layers for a list of linear layers
    public static func wrapLayers(
        _ layers: [(name: String, linear: Linear)],
        rank: Int,
        alpha: Float,
        dropoutRate: Float = 0.0
    ) -> [(name: String, lora: TrainableLoRALinear)] {
        layers.map { (name, linear) in
            let lora = TrainableLoRALinear(
                wrapping: linear,
                rank: rank,
                alpha: alpha,
                dropoutRate: dropoutRate
            )
            return (name, lora)
        }
    }
}

// MARK: - Training Mode

/// Thread-safe training mode state
public final class TrainingMode: @unchecked Sendable {
    public static let shared = TrainingMode()

    private var _isTraining: Bool = false
    private let lock = NSLock()

    private init() {}

    /// Whether training mode is enabled (affects dropout, etc.)
    public var isTraining: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _isTraining
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _isTraining = newValue
        }
    }
}
