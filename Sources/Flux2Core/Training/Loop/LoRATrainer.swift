// LoRATrainer.swift - Main LoRA training loop
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Main trainer class for LoRA fine-tuning
public final class LoRATrainer: @unchecked Sendable {
    
    // MARK: - Properties
    
    /// Training configuration
    public let config: LoRATrainingConfig
    
    /// Target model type
    public let modelType: Flux2Model
    
    /// Training dataset
    private var dataset: TrainingDataset?
    
    /// Latent cache
    private var latentCache: LatentCache?
    
    /// Text embedding cache
    private var textEmbeddingCache: TextEmbeddingCache?
    
    /// Reference to transformer with injected LoRA (for saving)
    private weak var loraTransformer: Flux2Transformer2DModel?
    
    /// Checkpoint manager
    private var checkpointManager: CheckpointManager?
    
    /// Training state
    private var state: TrainingState?
    
    /// Learning rate scheduler
    private var lrScheduler: LearningRateScheduler?

    /// AdamW optimizer for LoRA parameters
    private var optimizer: AdamW?

    /// Event handler
    private weak var eventHandler: TrainingEventHandler?
    
    /// Progress callback
    private var progressCallback: TrainingProgressCallback?
    
    /// Whether training is running
    private var isRunning: Bool = false
    
    /// Whether training should stop
    private var shouldStop: Bool = false

    /// Early stopping state
    private var bestLoss: Float = Float.infinity
    private var epochsWithoutImprovement: Int = 0
    private var earlyStopReason: String?

    /// Cached valueAndGrad function to avoid reconstructing gradient graph per step
    /// Type: (Model, [MLXArray]) -> ([MLXArray], ModuleParameters)
    private var cachedLossAndGrad: ((Flux2Transformer2DModel, [MLXArray]) -> ([MLXArray], ModuleParameters))?
    
    // MARK: - Initialization
    
    /// Initialize LoRA trainer
    /// - Parameters:
    ///   - config: Training configuration
    ///   - modelType: Target model type
    public init(config: LoRATrainingConfig, modelType: Flux2Model) {
        self.config = config
        self.modelType = modelType
    }
    
    /// Set event handler
    public func setEventHandler(_ handler: TrainingEventHandler) {
        self.eventHandler = handler
    }
    
    /// Set progress callback
    public func setProgressCallback(_ callback: @escaping TrainingProgressCallback) {
        self.progressCallback = callback
    }
    
    // MARK: - Training Setup
    
    /// Prepare for training
    public func prepare() async throws {
        Flux2Debug.log("[LoRATrainer] Preparing for training...")
        
        // Validate configuration
        try config.validate()
        
        // Initialize dataset
        dataset = try TrainingDataset(config: config)
        guard let dataset = dataset else {
            throw LoRATrainerError.datasetLoadFailed
        }
        
        let validation = dataset.validate()
        if !validation.isValid {
            throw LoRATrainerError.invalidDataset(validation.errors.joined(separator: ", "))
        }
        
        Flux2Debug.log("[LoRATrainer] Dataset loaded: \(dataset.count) samples")
        
        // Initialize caches
        if config.cacheLatents {
            latentCache = LatentCache(config: config)
        }
        
        if config.cacheTextEmbeddings {
            let textCacheDir = config.datasetPath.appendingPathComponent(".text_cache")
            textEmbeddingCache = TextEmbeddingCache(cacheDirectory: textCacheDir)
        }

        // Initialize checkpoint manager
        let outputDir = config.outputPath.deletingLastPathComponent()
        checkpointManager = CheckpointManager(
            outputDirectory: outputDir,
            maxCheckpoints: config.keepOnlyLastNCheckpoints
        )
        
        // Calculate total steps
        let stepsPerEpoch = dataset.batchesPerEpoch
        let totalSteps = config.maxSteps ?? (stepsPerEpoch * config.epochs)
        
        // Initialize training state
        state = TrainingState(
            totalSteps: totalSteps,
            totalEpochs: config.epochs
        )
        
        // Initialize learning rate scheduler
        lrScheduler = LRSchedulerFactory.create(
            type: config.lrScheduler,
            baseLR: config.learningRate,
            warmupSteps: config.warmupSteps,
            totalSteps: totalSteps
        )
        
        Flux2Debug.log("[LoRATrainer] Preparation complete")
        Flux2Debug.log("  Total steps: \(totalSteps)")
        Flux2Debug.log("  Steps per epoch: \(stepsPerEpoch)")
        
        // Memory estimation
        let estimatedMemory = config.estimateMemoryGB(for: modelType)
        Flux2Debug.log("  Estimated memory: \(String(format: "%.1f", estimatedMemory)) GB")
    }
    
    // MARK: - Pre-caching
    
    /// Pre-cache latents using VAE
    public func preCacheLatents(vae: AutoencoderKLFlux2) async throws {
        guard let dataset = dataset, let cache = latentCache else {
            throw LoRATrainerError.notPrepared
        }
        
        Flux2Debug.log("[LoRATrainer] Pre-caching latents...")
        
        try await cache.preEncodeDataset(dataset, vae: vae) { current, total in
            let progress = Float(current) / Float(total) * 100
            Flux2Debug.log("  Pre-caching: \(current)/\(total) (\(Int(progress))%)")
        }
        
        let stats = cache.getStatistics()
        Flux2Debug.log("[LoRATrainer] Latent caching complete:")
        Flux2Debug.log(stats.summary)
    }
    
    // MARK: - Main Training Loop
    
    /// Run the training loop
    /// - Parameters:
    ///   - transformer: The transformer model (will have LoRA injected)
    ///   - vae: VAE encoder (if not using cached latents)
    ///   - textEncoder: Function to encode text prompts to hidden states
    public func train(
        transformer: Flux2Transformer2DModel,
        vae: AutoencoderKLFlux2?,
        textEncoder: @escaping (String) async throws -> MLXArray
    ) async throws {
        guard let dataset = dataset,
              var state = state,
              let lrScheduler = lrScheduler,
              let _ = checkpointManager else {
            throw LoRATrainerError.notPrepared
        }

        isRunning = true
        shouldStop = false

        // Inject LoRA directly into transformer (replaces Linear with LoRAInjectedLinear)
        // This is the correct approach - LoRA becomes part of the forward pass
        transformer.applyLoRA(
            rank: config.rank,
            alpha: config.alpha,
            targetBlocks: config.targetLayers.toTargetBlocks()
        )
        loraTransformer = transformer

        Flux2Debug.log("[LoRATrainer] Injected LoRA into transformer with \(transformer.loraParameterCount) trainable parameters")

        // CRITICAL: Freeze base model weights so valueAndGrad only computes gradients for LoRA params
        // This dramatically speeds up training by not computing gradients for frozen weights
        transformer.freeze(recursive: true)

        // Unfreeze only LoRA parameters so they receive gradients
        transformer.unfreezeLoRAParameters()

        Flux2Debug.log("[LoRATrainer] Base model frozen, LoRA parameters unfrozen")

        // Create AdamW optimizer for the transformer's trainable parameters (LoRA only)
        self.optimizer = AdamW(
            learningRate: config.learningRate,
            betas: (config.adamBeta1, config.adamBeta2),
            eps: config.adamEpsilon,
            weightDecay: config.weightDecay
        )
        Flux2Debug.log("[LoRATrainer] Created AdamW optimizer (lr=\(config.learningRate), wd=\(config.weightDecay))")

        // Create cached valueAndGrad function ONCE - this is critical for performance
        // Creating it inside trainStep() forces gradient graph reconstruction per call
        let usesGuidance = modelType.usesGuidanceEmbeds
        func lossFunction(model: Flux2Transformer2DModel, arrays: [MLXArray]) -> [MLXArray] {
            // Input arrays order: [packedLatents, batchedHidden, timesteps, imgIds, txtIds, velocityTarget, guidance]
            let packedLatentsIn = arrays[0]
            let batchedHiddenIn = arrays[1]
            let timestepsIn = arrays[2]
            let imgIdsIn = arrays[3]
            let txtIdsIn = arrays[4]
            let velocityTargetIn = arrays[5]
            let guidanceIn: MLXArray? = usesGuidance ? arrays[6] : nil

            let modelOutput = model(
                hiddenStates: packedLatentsIn,
                encoderHiddenStates: batchedHiddenIn,
                timestep: timestepsIn,
                guidance: guidanceIn,
                imgIds: imgIdsIn,
                txtIds: txtIdsIn
            )

            // MSE loss between predicted velocity and target velocity
            let loss = mseLoss(predictions: modelOutput, targets: velocityTargetIn, reduction: .mean)
            return [loss]
        }
        self.cachedLossAndGrad = valueAndGrad(model: transformer, lossFunction)
        Flux2Debug.log("[LoRATrainer] Created cached valueAndGrad function for efficient gradient computation")

        // Set training mode
        TrainingMode.shared.isTraining = true
        
        // Emit start event
        eventHandler?.handleEvent(.started)
        
        // Resume from checkpoint if specified
        // TODO: Update checkpoint loading for LoRATrainingModel
        if config.resumeFromCheckpoint != nil {
            Flux2Debug.log("[LoRATrainer] Warning: Checkpoint resume not yet supported with new training model")
        }
        
        self.state = state
        
        do {
            // Training loop
            while !state.isComplete && !shouldStop {
                // Start new epoch if needed
                if state.epochStep == 0 {
                    state.startEpoch()
                    dataset.startEpoch()
                    eventHandler?.handleEvent(.epochStarted(epoch: state.epoch))
                }

                // Process batches
                while let batch = try dataset.nextBatch(), !shouldStop {
                    // Get current learning rate and update optimizer
                    let currentLR = lrScheduler.getLearningRate(step: state.globalStep)
                    optimizer?.learningRate = currentLR

                    // Training step
                    let loss = try await trainStep(
                        batch: batch,
                        transformer: transformer,
                        vae: vae,
                        textEncoder: textEncoder,
                        learningRate: currentLR
                    )
                    
                    // Update state
                    state.update(loss: loss, batchSize: batch.count)
                    self.state = state
                    
                    // Emit step event
                    eventHandler?.handleEvent(.stepCompleted(step: state.globalStep, loss: loss))
                    progressCallback?(state)
                    
                    // Logging
                    if state.globalStep % config.logEveryNSteps == 0 {
                        Flux2Debug.log(state.progressSummary)
                    }
                    
                    // Checkpointing
                    // TODO: Implement checkpoint saving with LoRATrainingModel
                    if config.saveEveryNSteps > 0 &&
                       state.globalStep % config.saveEveryNSteps == 0 {
                        // For now, skip intermediate checkpoints
                        state.lastCheckpointTime = Date()
                    }
                    
                    // Validation
                    if config.validationEveryNSteps > 0 &&
                       config.validationPrompt != nil &&
                       state.globalStep % config.validationEveryNSteps == 0 {
                        // TODO: Generate validation image
                        state.lastValidationTime = Date()
                    }
                    
                    // Check completion
                    if state.isComplete {
                        break
                    }
                }
                
                // Epoch complete
                eventHandler?.handleEvent(.epochCompleted(
                    epoch: state.epoch,
                    avgLoss: state.averageLoss
                ))

                // Early stopping check
                if config.enableEarlyStopping {
                    let currentLoss = state.averageLoss
                    let improvement = bestLoss - currentLoss

                    if improvement > config.earlyStoppingMinDelta {
                        // Loss improved
                        bestLoss = currentLoss
                        epochsWithoutImprovement = 0
                        Flux2Debug.log("[LoRATrainer] Early stopping: loss improved to \(String(format: "%.4f", currentLoss))")
                    } else {
                        // No significant improvement
                        epochsWithoutImprovement += 1
                        Flux2Debug.log("[LoRATrainer] Early stopping: no improvement for \(epochsWithoutImprovement)/\(config.earlyStoppingPatience) epochs")

                        if epochsWithoutImprovement >= config.earlyStoppingPatience {
                            earlyStopReason = "Loss plateau detected (best: \(String(format: "%.4f", bestLoss)), current: \(String(format: "%.4f", currentLoss)))"
                            Flux2Debug.log("[LoRATrainer] Early stopping triggered: \(earlyStopReason!)")
                            shouldStop = true
                        }
                    }
                }

                // Reset epoch step for next epoch
                if !state.isComplete && !shouldStop {
                    state.epochStep = 0
                    self.state = state
                }
            }
            
            // Training complete
            TrainingMode.shared.isTraining = false
            isRunning = false
            
            // Save final LoRA weights from transformer
            try saveLoRAWeights(from: transformer, to: config.outputPath)
            
            eventHandler?.handleEvent(.completed(
                finalLoss: state.averageLoss,
                totalSteps: state.globalStep
            ))
            
            Flux2Debug.log("[LoRATrainer] Training complete!")
            Flux2Debug.log(state.detailedStatus)
            
        } catch {
            TrainingMode.shared.isTraining = false
            isRunning = false
            eventHandler?.handleEvent(.error(error))
            throw error
        }
    }
    
    // MARK: - Single Training Step

    /// Struct to hold all inputs for the loss function
    private struct TrainingInputs {
        let packedLatents: MLXArray
        let batchedHidden: MLXArray
        let timesteps: MLXArray
        let guidance: MLXArray?
        let imgIds: MLXArray
        let txtIds: MLXArray
        /// Velocity target for flow matching: v = noise - original_latents
        let packedVelocityTarget: MLXArray
    }

    /// Execute a single training step with real gradient computation
    private func trainStep(
        batch: TrainingBatch,
        transformer: Flux2Transformer2DModel,
        vae: AutoencoderKLFlux2?,
        textEncoder: (String) async throws -> MLXArray,
        learningRate: Float
    ) async throws -> Float {
        // Get latents (from cache or encode)
        let latents: MLXArray
        if let cache = latentCache {
            latents = try cache.getLatents(for: batch, vae: vae)
        } else if let vae = vae {
            // Encode images on the fly
            // batch.images is [B, H, W, C], VAE expects [B, C, H, W]
            let normalizedImages = batch.images * 2.0 - 1.0
            let nchwImages = normalizedImages.transposed(0, 3, 1, 2)  // NHWC -> NCHW
            latents = vae.encode(nchwImages)
        } else {
            throw LoRATrainerError.noVAEProvided
        }

        // Get text embeddings (from cache or encode)
        var hiddenStates: [MLXArray] = []

        for caption in batch.captions {
            if let cache = textEmbeddingCache,
               let cached = try cache.getEmbeddings(for: caption) {
                // Squeeze batch dimension if present: [1, seq, dim] -> [seq, dim]
                let embedding = cached.hidden.shape.count > 2
                    ? cached.hidden.squeezed(axis: 0)
                    : cached.hidden
                hiddenStates.append(embedding)
            } else {
                let embeddings = try await textEncoder(caption)
                // Squeeze batch dimension if present: [1, seq, dim] -> [seq, dim]
                let embedding = embeddings.shape.count > 2
                    ? embeddings.squeezed(axis: 0)
                    : embeddings
                hiddenStates.append(embedding)

                // Cache for next time (Flux2 uses hidden states only, pooled is a placeholder)
                if let cache = textEmbeddingCache {
                    try cache.saveEmbeddings(
                        pooled: MLXArray.zeros([1]),  // Placeholder
                        hidden: embedding,
                        for: caption
                    )
                }
            }
        }

        // Stack embeddings: [[seq, dim], ...] -> [B, seq, dim]
        let batchedHidden = MLX.stacked(hiddenStates, axis: 0)

        // Create 4D position IDs for Flux2 RoPE
        let batchSize = latents.shape[0]
        let patchSize = 2  // Standard patch size for Flux
        let imgH = latents.shape[2] / patchSize
        let imgW = latents.shape[3] / patchSize
        let txtLen = batchedHidden.shape[1]

        // Generate proper 4D position IDs for RoPE [seq_len, 4]
        let txtIds = generateTextPositionIDs(length: txtLen)
        let imgIds = generateImagePositionIDs(height: imgH, width: imgW)

        // Sample timesteps
        let timesteps = MLXRandom.randInt(low: 0, high: 1000, [batchSize])

        // Get sigmas - for flow matching, sigma = t directly (no transformation)
        // The model is trained to predict velocity v = noise - latents
        // at noise level sigma, where x_t = (1 - sigma) * x_0 + sigma * noise
        let sigmas = timesteps.asType(.float32) / 1000.0

        // Sample noise
        let noise = MLXRandom.normal(latents.shape)

        // Add noise to latents
        let sigmasExpanded = sigmas.reshaped([batchSize, 1, 1, 1])
        let noisyLatents = (1 - sigmasExpanded) * latents + sigmasExpanded * noise

        // Pack latents for transformer: [B, C, H, W] -> [B, H*W/4, C*4]
        let packedLatents = packLatentsForTransformer(noisyLatents, patchSize: 2)

        // Prepare guidance
        let guidance: MLXArray? = modelType.usesGuidanceEmbeds ?
            MLXArray(Array(repeating: Float(4.0), count: batchSize)) : nil

        // Compute velocity target for flow matching: v = noise - original_latents
        // Flux.2 uses rectified flow matching where the model predicts the velocity
        // from original latents (x_0) to noise (x_1)
        let velocityTarget = noise - latents
        let packedVelocityTarget = packLatentsForTransformer(velocityTarget, patchSize: 2)

        // Prepare inputs as array of MLXArray for efficient valueAndGrad caching
        // Order: [packedLatents, batchedHidden, timesteps, imgIds, txtIds, packedVelocityTarget, guidance(optional)]
        var inputArrays: [MLXArray] = [
            packedLatents,
            batchedHidden,
            timesteps.asType(DType.float32),
            imgIds.asType(DType.int32),
            txtIds.asType(DType.int32),
            packedVelocityTarget
        ]
        // Add guidance as 7th element if present, otherwise add a dummy zero array
        if let g = guidance {
            inputArrays.append(g)
        } else {
            inputArrays.append(MLXArray(0.0))  // Dummy placeholder
        }

        // Use the cached valueAndGrad function created in train()
        // This is critical for performance - creating it inside trainStep forces gradient graph reconstruction
        guard let lossAndGrad = self.cachedLossAndGrad else {
            throw LoRATrainerError.trainingFailed("cachedLossAndGrad not initialized")
        }

        // Compute loss and gradients using cached function
        let (losses, grads) = lossAndGrad(transformer, inputArrays)
        let loss = losses[0]

        // Filter gradients to keep only LoRA parameters (loraA and loraB)
        let filteredGrads = filterLoRAGradients(grads)

        // Apply gradient clipping if configured
        var clippedGrads = filteredGrads
        if config.maxGradNorm > 0 {
            clippedGrads = clipGradNorm(filteredGrads, maxNorm: config.maxGradNorm)
        }

        // Update LoRA parameters using AdamW optimizer
        guard let optimizer = self.optimizer else {
            throw LoRATrainerError.trainingFailed("Optimizer not initialized")
        }
        optimizer.update(model: transformer, gradients: clippedGrads)

        // IMPORTANT: Must call eval() at EVERY step to prevent MLX lazy evaluation graph accumulation
        // Without eval(), the computation graph grows exponentially, causing later steps to be extremely slow
        // Note: The idea of calling eval() every N steps doesn't work with MLX's lazy evaluation model
        eval(transformer, optimizer, loss)
        let lossValue = loss.item(Float.self)

        // Clear GPU cache periodically
        let currentStep = (state?.globalStep ?? 0) + 1
        if currentStep % 10 == 0 {
            MLX.Memory.clearCache()
        }

        return lossValue
    }

    /// Flatten ModuleParameters gradients into a simple [path: MLXArray] dictionary
    private func flattenGradients(_ grads: ModuleParameters, prefix: String = "") -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        func flatten(_ item: NestedItem<String, MLXArray>, path: String) {
            switch item {
            case .none:
                break
            case .value(let arr):
                result[path] = arr
            case .array(let items):
                for (idx, subItem) in items.enumerated() {
                    flatten(subItem, path: "\(path)[\(idx)]")
                }
            case .dictionary(let dict):
                for (key, subItem) in dict {
                    let newPath = path.isEmpty ? key : "\(path).\(key)"
                    flatten(subItem, path: newPath)
                }
            }
        }

        for (key, item) in grads {
            flatten(item, path: key)
        }

        return result
    }

    /// Clip gradient norm using flattened gradient dictionary
    private func clipFlatGradNorm(_ grads: [String: MLXArray], maxNorm: Float) -> [String: MLXArray] {
        guard !grads.isEmpty else { return grads }

        // Compute total norm
        var totalNormSq = MLXArray(0.0)
        for (_, grad) in grads {
            totalNormSq = totalNormSq + (grad * grad).sum()
        }
        let totalNorm = sqrt(totalNormSq)

        // Compute clip coefficient
        let maxNormArr = MLXArray(maxNorm)
        let clipCoef = minimum(maxNormArr / (totalNorm + MLXArray(1e-6)), MLXArray(1.0))

        // Apply clipping
        var result: [String: MLXArray] = [:]
        for (key, grad) in grads {
            result[key] = grad * clipCoef
        }

        return result
    }

    /// Clip gradient norm to prevent exploding gradients (legacy, kept for reference)
    private func clipGradNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        // Flatten all gradients and compute total norm
        var allGrads: [MLXArray] = []

        func collectGrads(_ item: NestedItem<String, MLXArray>) {
            switch item {
            case .none:
                break
            case .value(let arr):
                allGrads.append(arr)
            case .array(let items):
                for subItem in items {
                    collectGrads(subItem)
                }
            case .dictionary(let dict):
                for (_, subItem) in dict {
                    collectGrads(subItem)
                }
            }
        }

        for (_, item) in grads {
            collectGrads(item)
        }

        // Compute total norm
        guard !allGrads.isEmpty else { return grads }

        var totalNormSq = MLXArray(0.0)
        for grad in allGrads {
            totalNormSq = totalNormSq + (grad * grad).sum()
        }
        let totalNorm = sqrt(totalNormSq)

        // Compute clip coefficient
        let maxNormArr = MLXArray(maxNorm)
        let clipCoef = minimum(maxNormArr / (totalNorm + MLXArray(1e-6)), MLXArray(1.0))

        // Apply clipping
        func clipItem(_ item: NestedItem<String, MLXArray>) -> NestedItem<String, MLXArray> {
            switch item {
            case .none:
                return .none
            case .value(let arr):
                return .value(arr * clipCoef)
            case .array(let items):
                return .array(items.map { clipItem($0) })
            case .dictionary(let dict):
                return .dictionary(dict.mapValues { clipItem($0) })
            }
        }

        var result = ModuleParameters()
        for (key, item) in grads {
            result[key] = clipItem(item)
        }

        return result
    }

    /// Debug: Print gradient paths containing "lora"
    private func printGradientPaths(_ grads: ModuleParameters, prefix: String, limit: Int) {
        var count = 0
        var loraCount = 0

        func printRecursive(_ item: NestedItem<String, MLXArray>, path: String) {
            switch item {
            case .none:
                break
            case .value(let arr):
                // Print if path contains "lora" or if we haven't printed many yet
                if path.lowercased().contains("lora") {
                    print("[DEBUG PATHS LORA] \(path): shape=\(arr.shape)")
                    loraCount += 1
                } else if count < limit {
                    print("[DEBUG PATHS] \(path): shape=\(arr.shape)")
                    count += 1
                }
            case .array(let items):
                for (idx, subItem) in items.enumerated() {
                    printRecursive(subItem, path: "\(path)[\(idx)]")
                }
            case .dictionary(let dict):
                for (key, subItem) in dict {
                    let newPath = path.isEmpty ? key : "\(path).\(key)"
                    printRecursive(subItem, path: newPath)
                }
            }
        }

        for (key, item) in grads {
            printRecursive(item, path: key)
        }
        print("[DEBUG PATHS] Total LoRA paths found: \(loraCount)")
        fflush(stdout)
    }

    /// Filter gradients to keep only LoRA parameters (loraA and loraB)
    /// Non-LoRA gradients are removed (.none) to prevent updates to base model weights
    /// This is much faster than zeroing out - we simply skip non-LoRA parameters
    private func filterLoRAGradients(_ grads: ModuleParameters) -> ModuleParameters {
        // Recursive function that properly tracks path and filters
        func filterRecursive(_ item: NestedItem<String, MLXArray>, path: [String]) -> NestedItem<String, MLXArray> {
            switch item {
            case .none:
                return .none
            case .value(let arr):
                // Check if the last path component is loraA or loraB
                let lastKey = path.last ?? ""
                if lastKey == "loraA" || lastKey == "loraB" {
                    return .value(arr)
                } else {
                    // Skip non-LoRA gradients entirely (much faster than creating zeros)
                    return .none
                }
            case .array(let items):
                let filteredItems = items.enumerated().map { (idx, subItem) in
                    filterRecursive(subItem, path: path + ["[\(idx)]"])
                }
                // Check if all items are .none, if so return .none
                let hasNonNone = filteredItems.contains { item in
                    if case .none = item { return false }
                    return true
                }
                return hasNonNone ? .array(filteredItems) : .none
            case .dictionary(let dict):
                var newDict: [String: NestedItem<String, MLXArray>] = [:]
                for (key, subItem) in dict {
                    let filtered = filterRecursive(subItem, path: path + [key])
                    // Only include non-.none items
                    if case .none = filtered {
                        continue
                    }
                    newDict[key] = filtered
                }
                return newDict.isEmpty ? .none : .dictionary(newDict)
            }
        }

        var result = ModuleParameters()
        for (key, item) in grads {
            let filtered = filterRecursive(item, path: [key])
            // Only include non-.none items
            if case .none = filtered {
                continue
            }
            result[key] = filtered
        }

        return result
    }

    // MARK: - LoRA Parameter Helpers

    /// Apply gradients to LoRA parameters in the transformer
    /// Uses simple SGD: param = param - lr * grad
    private func applyLoRAGradients(
        transformer: Flux2Transformer2DModel,
        gradients: ModuleParameters,
        learningRate: Float
    ) {
        // Flatten gradients for easier access
        let flatGrads = flattenGradients(gradients)

        // Debug: log gradient paths on first step
        if state?.globalStep == 1 {
            Flux2Debug.log("[ApplyLoRAGradients] Available gradient paths (\(flatGrads.count) total):")
            for path in flatGrads.keys.sorted().prefix(20) {
                Flux2Debug.log("  - \(path)")
            }
            if flatGrads.count > 20 {
                Flux2Debug.log("  ... and \(flatGrads.count - 20) more")
            }
        }

        // Update double-stream blocks
        for (idx, block) in transformer.transformerBlocks.enumerated() {
            // Update each attention projection if it's a LoRAInjectedLinear
            updateLoRAIfPresent(block.attn.toQ, gradPath: "transformerBlocks[\(idx)].attn.toQ", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toK, gradPath: "transformerBlocks[\(idx)].attn.toK", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toV, gradPath: "transformerBlocks[\(idx)].attn.toV", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.addQProj, gradPath: "transformerBlocks[\(idx)].attn.addQProj", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.addKProj, gradPath: "transformerBlocks[\(idx)].attn.addKProj", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.addVProj, gradPath: "transformerBlocks[\(idx)].attn.addVProj", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toOut, gradPath: "transformerBlocks[\(idx)].attn.toOut", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toAddOut, gradPath: "transformerBlocks[\(idx)].attn.toAddOut", flatGrads: flatGrads, lr: learningRate)
        }

        // Update single-stream blocks
        for (idx, block) in transformer.singleTransformerBlocks.enumerated() {
            updateLoRAIfPresent(block.attn.toQkvMlp, gradPath: "singleTransformerBlocks[\(idx)].attn.toQkvMlp", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toOut, gradPath: "singleTransformerBlocks[\(idx)].attn.toOut", flatGrads: flatGrads, lr: learningRate)
        }
    }

    /// Update a single LoRA layer if it exists and has gradients
    private func updateLoRAIfPresent(_ linear: Linear, gradPath: String, flatGrads: [String: MLXArray], lr: Float) {
        guard let lora = linear as? LoRAInjectedLinear else { return }

        // Convert bracket notation to dot notation for alternative matching
        // e.g., "transformerBlocks[0]" -> "transformerBlocks.0"
        let dotPath = gradPath.replacingOccurrences(of: "[", with: ".").replacingOccurrences(of: "]", with: "")

        // Try different possible gradient paths (bracket and dot notation)
        let possiblePaths = [
            "\(gradPath).lora_a",
            "\(dotPath).lora_a",
            "\(gradPath).loraA",
            "\(dotPath).loraA"
        ]

        var gradA: MLXArray? = nil
        var gradB: MLXArray? = nil

        for basePath in possiblePaths {
            if gradA == nil, let g = flatGrads[basePath] {
                gradA = g
            }
            let bPath = basePath.replacingOccurrences(of: "lora_a", with: "lora_b")
                                .replacingOccurrences(of: "loraA", with: "loraB")
            if gradB == nil, let g = flatGrads[bPath] {
                gradB = g
            }
        }

        // Apply SGD update
        if let gA = gradA {
            lora.loraA = lora.loraA - lr * gA
        }
        if let gB = gradB {
            lora.loraB = lora.loraB - lr * gB
        }

        // Debug warning if no gradients found
        if gradA == nil && gradB == nil && state?.globalStep == 1 {
            Flux2Debug.log("[ApplyLoRAGradients] WARNING: No gradients found for path \(gradPath)")
        }
    }

    /// Evaluate all LoRA parameters to synchronize GPU
    private func evalLoRAParameters(transformer: Flux2Transformer2DModel) {
        // Double-stream blocks
        for block in transformer.transformerBlocks {
            if let lora = block.attn.toQ as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toK as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toV as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
        }

        // Single-stream blocks
        for block in transformer.singleTransformerBlocks {
            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
        }
    }

    // MARK: - Save Weights

    /// Save LoRA weights from transformer to safetensors file
    private func saveLoRAWeights(from transformer: Flux2Transformer2DModel, to url: URL) throws {
        // Get LoRA weights from transformer (already in Diffusers format)
        let weights = transformer.getLoRAParameters()

        guard !weights.isEmpty else {
            throw LoRATrainerError.trainingFailed("No LoRA weights found in transformer")
        }

        // Save to safetensors format
        try save(arrays: weights, url: url)

        // Also save metadata alongside
        let metadataPath = url.deletingPathExtension().appendingPathExtension("json")
        let metadata = LoRAWeightsMetadata(
            rank: config.rank,
            alpha: config.alpha,
            targetLayers: config.targetLayers.rawValue,
            triggerWord: config.triggerWord,
            trainedOn: Date()
        )
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: metadataPath)

        Flux2Debug.log("[LoRATrainer] Saved LoRA weights to \(url.path) (\(weights.count) tensors)")
    }

    // MARK: - Control

    /// Stop training gracefully
    public func stop() {
        shouldStop = true
        Flux2Debug.log("[LoRATrainer] Stopping training...")
    }
    
    /// Get current training state
    public var currentState: TrainingState? {
        state
    }
    
    /// Whether training is currently running
    public var running: Bool {
        isRunning
    }

    // MARK: - Helpers

    /// Pack latents into patches for transformer input
    /// - Parameters:
    ///   - latents: Latents in [B, C, H, W] format
    ///   - patchSize: Size of each patch (typically 2)
    /// - Returns: Packed latents in [B, seq_len, patch_features] format
    private func packLatentsForTransformer(_ latents: MLXArray, patchSize: Int) -> MLXArray {
        let shape = latents.shape
        let batchSize = shape[0]
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]

        let patchH = height / patchSize
        let patchW = width / patchSize
        let patchFeatures = channels * patchSize * patchSize

        // Reshape: [B, C, H, W] -> [B, C, patchH, patchSize, patchW, patchSize]
        let reshaped = latents.reshaped([batchSize, channels, patchH, patchSize, patchW, patchSize])

        // Transpose to group spatial patches: [B, patchH, patchW, C, patchSize, patchSize]
        let transposed = reshaped.transposed(0, 2, 4, 1, 3, 5)

        // Flatten to sequence: [B, patchH * patchW, C * patchSize * patchSize]
        let packed = transposed.reshaped([batchSize, patchH * patchW, patchFeatures])

        return packed
    }
}

// MARK: - Errors

public enum LoRATrainerError: Error, LocalizedError {
    case notPrepared
    case datasetLoadFailed
    case invalidDataset(String)
    case noVAEProvided
    case trainingFailed(String)
    case checkpointLoadFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .notPrepared:
            return "Trainer not prepared. Call prepare() first."
        case .datasetLoadFailed:
            return "Failed to load training dataset"
        case .invalidDataset(let reason):
            return "Invalid dataset: \(reason)"
        case .noVAEProvided:
            return "No VAE provided and latents not cached"
        case .trainingFailed(let reason):
            return "Training failed: \(reason)"
        case .checkpointLoadFailed(let reason):
            return "Failed to load checkpoint: \(reason)"
        }
    }
}
