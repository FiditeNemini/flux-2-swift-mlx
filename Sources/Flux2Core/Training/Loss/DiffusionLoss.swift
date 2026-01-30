// DiffusionLoss.swift - Loss functions for diffusion model training
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXRandom

/// Loss type for diffusion training
public enum DiffusionLossType: String, Codable, Sendable {
    /// Predict the noise that was added
    case noise = "noise"
    
    /// Predict the velocity (v-prediction)
    case velocity = "velocity"
    
    /// Predict the original sample (epsilon prediction variant)
    case sample = "sample"
}

/// Loss weighting scheme
public enum LossWeighting: String, Codable, Sendable {
    /// No weighting (uniform across timesteps)
    case uniform = "uniform"
    
    /// Min-SNR weighting (better convergence)
    case minSNR = "min_snr"
    
    /// Cosine weighting
    case cosine = "cosine"
    
    /// Sigmoid weighting
    case sigmoid = "sigmoid"
}

/// Diffusion loss calculator for LoRA training
public struct DiffusionLoss: Sendable {
    
    /// Loss type (noise, velocity, or sample prediction)
    public let lossType: DiffusionLossType
    
    /// Loss weighting scheme
    public let weighting: LossWeighting
    
    /// Min-SNR gamma (only used if weighting = .minSNR)
    public let snrGamma: Float
    
    /// Initialize diffusion loss
    /// - Parameters:
    ///   - lossType: Type of prediction loss
    ///   - weighting: Loss weighting scheme
    ///   - snrGamma: Gamma for min-SNR weighting (default 5.0)
    public init(
        lossType: DiffusionLossType = .noise,
        weighting: LossWeighting = .uniform,
        snrGamma: Float = 5.0
    ) {
        self.lossType = lossType
        self.weighting = weighting
        self.snrGamma = snrGamma
    }
    
    // MARK: - Loss Computation
    
    /// Compute diffusion training loss
    /// - Parameters:
    ///   - modelOutput: Output from the transformer
    ///   - target: Target (noise, velocity, or sample depending on lossType)
    ///   - timesteps: Timesteps used for each sample [B]
    ///   - sigmas: Sigma values for each sample [B] (optional, for weighting)
    /// - Returns: Scalar loss value
    public func compute(
        modelOutput: MLXArray,
        target: MLXArray,
        timesteps: MLXArray,
        sigmas: MLXArray? = nil
    ) -> MLXArray {
        // MSE loss per sample
        let diff = modelOutput - target
        let squaredDiff = diff * diff
        
        // Mean over spatial and channel dimensions, keep batch
        let mseLoss = MLX.mean(squaredDiff, axes: [1, 2, 3])
        
        // Apply weighting
        let weights = computeWeights(timesteps: timesteps, sigmas: sigmas)
        let weightedLoss = mseLoss * weights
        
        // Mean over batch
        return MLX.mean(weightedLoss)
    }
    
    /// Compute simple MSE loss without weighting
    public func simpleMSE(
        modelOutput: MLXArray,
        target: MLXArray
    ) -> MLXArray {
        let diff = modelOutput - target
        return MLX.mean(diff * diff)
    }
    
    /// Compute loss weights based on timesteps
    private func computeWeights(
        timesteps: MLXArray,
        sigmas: MLXArray?
    ) -> MLXArray {
        switch weighting {
        case .uniform:
            return MLXArray.ones([timesteps.shape[0]])
            
        case .minSNR:
            guard let sigmas = sigmas else {
                return MLXArray.ones([timesteps.shape[0]])
            }
            return minSNRWeights(sigmas: sigmas)
            
        case .cosine:
            return cosineWeights(timesteps: timesteps)
            
        case .sigmoid:
            return sigmoidWeights(timesteps: timesteps)
        }
    }
    
    /// Min-SNR weighting: min(SNR, gamma) / SNR
    private func minSNRWeights(sigmas: MLXArray) -> MLXArray {
        // SNR = 1 / sigma^2
        let snr = 1.0 / (sigmas * sigmas + 1e-8)
        
        // Clamp SNR
        let clampedSNR = MLX.minimum(snr, MLXArray(snrGamma))
        
        // Weight = clampedSNR / SNR
        return clampedSNR / (snr + 1e-8)
    }
    
    /// Cosine weighting based on timestep
    private func cosineWeights(timesteps: MLXArray) -> MLXArray {
        // Normalize timesteps to [0, 1]
        let t = timesteps.asType(.float32) / 1000.0
        
        // Cosine schedule: cos(t * pi / 2)^2
        let cosWeights = MLX.cos(t * Float.pi / 2)
        return cosWeights * cosWeights
    }
    
    /// Sigmoid weighting based on timestep
    private func sigmoidWeights(timesteps: MLXArray) -> MLXArray {
        // Normalize to [-5, 5] range
        let t = (timesteps.asType(.float32) / 1000.0 - 0.5) * 10
        
        // Sigmoid: 1 / (1 + exp(-t))
        return MLX.sigmoid(-t)  // Inverted so early timesteps have higher weight
    }
    
    // MARK: - Target Computation
    
    /// Get the training target based on loss type
    /// - Parameters:
    ///   - noise: The noise that was added
    ///   - sample: The original clean sample
    ///   - noisySample: The noisy sample
    ///   - sigma: The noise level
    /// - Returns: The target for the model to predict
    public func getTarget(
        noise: MLXArray,
        sample: MLXArray,
        noisySample: MLXArray,
        sigma: MLXArray
    ) -> MLXArray {
        switch lossType {
        case .noise:
            return noise
            
        case .sample:
            return sample
            
        case .velocity:
            // v = sigma * sample - (1 - sigma) * noise
            // Reshape sigma for broadcasting
            let sigmaExpanded = sigma.reshaped([-1, 1, 1, 1])
            return sigmaExpanded * sample - (1 - sigmaExpanded) * noise
        }
    }
    
    /// Convert model output to noise prediction
    public func toNoisePrediction(
        modelOutput: MLXArray,
        sample: MLXArray,
        sigma: MLXArray
    ) -> MLXArray {
        switch lossType {
        case .noise:
            return modelOutput
            
        case .sample:
            // noise = (noisy_sample - output) / sigma
            let sigmaExpanded = sigma.reshaped([-1, 1, 1, 1])
            return (sample - modelOutput) / (sigmaExpanded + 1e-8)
            
        case .velocity:
            // From v-prediction: noise = sigma * v + sqrt(1 - sigma^2) * sample
            let sigmaExpanded = sigma.reshaped([-1, 1, 1, 1])
            let sqrtOneMinusSigmaSq = MLX.sqrt(1 - sigmaExpanded * sigmaExpanded + 1e-8)
            return sigmaExpanded * modelOutput + sqrtOneMinusSigmaSq * sample
        }
    }
}

// MARK: - Training Step Loss

/// Complete loss computation for a training step
public struct TrainingStepLoss: Sendable {
    
    /// The diffusion loss calculator
    public let diffusionLoss: DiffusionLoss
    
    /// Number of training timesteps
    public let numTrainTimesteps: Int
    
    /// Shift parameter for timestep sampling
    public let shift: Float
    
    /// Initialize training step loss
    public init(
        diffusionLoss: DiffusionLoss = DiffusionLoss(),
        numTrainTimesteps: Int = 1000,
        shift: Float = 3.0
    ) {
        self.diffusionLoss = diffusionLoss
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
    }
    
    /// Sample random timesteps for training
    public func sampleTimesteps(batchSize: Int) -> MLXArray {
        // Uniform sampling from [0, numTrainTimesteps)
        let timesteps = MLXRandom.randInt(
            low: 0,
            high: numTrainTimesteps,
            [batchSize]
        )
        return timesteps
    }
    
    /// Get sigma from timestep using flow matching schedule
    public func getSigma(timestep: MLXArray) -> MLXArray {
        // Normalize to [0, 1]
        let t = timestep.asType(.float32) / Float(numTrainTimesteps)
        
        // Apply shift (flow matching scheduler style)
        let sigma = t * shift / (1 + (shift - 1) * t)
        
        return sigma
    }
    
    /// Add noise to samples
    public func addNoise(
        samples: MLXArray,
        noise: MLXArray,
        sigma: MLXArray
    ) -> MLXArray {
        // Expand sigma for broadcasting [B] -> [B, 1, 1, 1]
        let sigmaExpanded = sigma.reshaped([-1, 1, 1, 1])
        
        // noisy = (1 - sigma) * sample + sigma * noise
        return (1 - sigmaExpanded) * samples + sigmaExpanded * noise
    }
    
    /// Compute full training loss
    /// - Parameters:
    ///   - model: Function that runs the model forward pass
    ///   - latents: Clean latent samples [B, H, W, C]
    ///   - textEmbeddings: Text embeddings for conditioning
    ///   - pooledEmbeddings: Pooled text embeddings
    /// - Returns: Scalar loss value
    public func computeLoss(
        modelForward: (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray,
        latents: MLXArray,
        textEmbeddings: MLXArray,
        pooledEmbeddings: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray
    ) -> MLXArray {
        let batchSize = latents.shape[0]
        
        // Sample timesteps
        let timesteps = sampleTimesteps(batchSize: batchSize)
        
        // Get sigmas
        let sigmas = getSigma(timestep: timesteps)
        
        // Sample noise
        let noise = MLXRandom.normal(latents.shape)
        
        // Add noise to latents
        let noisyLatents = addNoise(samples: latents, noise: noise, sigma: sigmas)
        
        // Get target based on loss type
        let target = diffusionLoss.getTarget(
            noise: noise,
            sample: latents,
            noisySample: noisyLatents,
            sigma: sigmas
        )
        
        // Model forward pass
        let modelOutput = modelForward(
            noisyLatents,
            textEmbeddings,
            timesteps.asType(.float32),
            imgIds,
            txtIds
        )
        
        // Compute loss
        return diffusionLoss.compute(
            modelOutput: modelOutput,
            target: target,
            timesteps: timesteps,
            sigmas: sigmas
        )
    }
}

// MARK: - Metrics

/// Training metrics for logging
public struct TrainingMetrics: Sendable {
    /// Current loss value
    public var loss: Float
    
    /// Moving average loss
    public var avgLoss: Float
    
    /// Current learning rate
    public var learningRate: Float
    
    /// Current step
    public var step: Int
    
    /// Current epoch
    public var epoch: Int
    
    /// Samples processed
    public var samplesProcessed: Int
    
    /// Gradient norm (if computed)
    public var gradNorm: Float?
    
    public init(
        loss: Float = 0,
        avgLoss: Float = 0,
        learningRate: Float = 0,
        step: Int = 0,
        epoch: Int = 0,
        samplesProcessed: Int = 0,
        gradNorm: Float? = nil
    ) {
        self.loss = loss
        self.avgLoss = avgLoss
        self.learningRate = learningRate
        self.step = step
        self.epoch = epoch
        self.samplesProcessed = samplesProcessed
        self.gradNorm = gradNorm
    }
    
    public var summary: String {
        var parts = [
            "step \(step)",
            "loss: \(String(format: "%.4f", loss))",
            "avg: \(String(format: "%.4f", avgLoss))",
            "lr: \(String(format: "%.2e", learningRate))"
        ]
        
        if let norm = gradNorm {
            parts.append("grad: \(String(format: "%.2f", norm))")
        }
        
        return parts.joined(separator: " | ")
    }
}
