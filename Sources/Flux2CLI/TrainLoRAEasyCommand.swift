// TrainLoRAEasyCommand.swift - Simplified LoRA training with auto-configuration
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import MLX

// MARK: - Easy Train LoRA Command

struct TrainLoRAEasy: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train-lora-easy",
        abstract: "Train a LoRA adapter with automatic configuration (simplified mode)"
    )

    // MARK: - Required Arguments

    @Argument(help: "Path to folder containing training images (PNG, JPG, WebP)")
    var imageFolder: String

    @Argument(help: "Trigger word for your LoRA (e.g., 'mystyle', 'sks', 'ohwx')")
    var triggerWord: String

    // MARK: - Optional Arguments

    @Option(name: .shortAndLong, help: "Output path for the trained LoRA (default: <trigger>.safetensors)")
    var output: String?

    @Option(name: .long, help: "Model to train: klein-4b, klein-9b (default: klein-4b)")
    var model: String = "klein-4b"

    @Option(name: .long, help: "Training intensity: light, normal, strong (default: normal)")
    var intensity: String = "normal"

    @Option(name: .long, help: "Subject type: person, style, object, character (helps with captioning)")
    var subject: String = "object"

    @Flag(name: .long, help: "Generate captions using VLM (requires more memory)")
    var autoCaptions: Bool = false

    @Flag(name: .long, help: "Show detailed progress")
    var verbose: Bool = false

    // MARK: - Run

    func run() async throws {
        print()
        print("=".repeating(60))
        print("  LoRA Easy Training Mode")
        print("=".repeating(60))
        print()

        // Validate model
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: klein-4b, klein-9b")
        }

        // Validate intensity
        guard let trainingIntensity = TrainingIntensity(rawValue: intensity) else {
            throw ValidationError("Invalid intensity: \(intensity). Use: light, normal, strong")
        }

        // Validate subject type
        guard let subjectType = SubjectType(rawValue: subject) else {
            throw ValidationError("Invalid subject: \(subject). Use: person, style, object, character")
        }

        // Validate image folder
        let folderURL = URL(fileURLWithPath: imageFolder)
        guard FileManager.default.fileExists(atPath: folderURL.path) else {
            throw ValidationError("Image folder not found: \(imageFolder)")
        }

        // Find images
        let imageExtensions = ["png", "jpg", "jpeg", "webp"]
        let contents = try FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
        let images = contents.filter { imageExtensions.contains($0.pathExtension.lowercased()) }

        guard !images.isEmpty else {
            throw ValidationError("No images found in folder. Supported formats: PNG, JPG, WebP")
        }

        print("Found \(images.count) images in \(folderURL.lastPathComponent)/")
        print("Trigger word: \"\(triggerWord)\"")
        print("Model: \(modelVariant.displayName)")
        print("Training intensity: \(trainingIntensity.displayName)")
        print()

        // Auto-generate or check captions
        print("Preparing captions...")
        let captionStats = try await prepareCaptoins(
            images: images,
            triggerWord: triggerWord,
            subjectType: subjectType,
            useVLM: autoCaptions
        )
        print("  \(captionStats.existing) existing, \(captionStats.generated) generated")
        print()

        // Calculate optimal parameters
        let params = calculateOptimalParameters(
            imageCount: images.count,
            intensity: trainingIntensity,
            model: modelVariant
        )

        print("Auto-configured parameters:")
        print("  Rank: \(params.rank)")
        print("  Max epochs: \(params.epochs)")
        print("  Learning rate: \(String(format: "%.1e", params.learningRate))")
        print("  Max steps: \(params.totalSteps)")
        print("  Estimated time: \(params.estimatedTime)")
        print("  Early stopping: enabled (patience=5 epochs)")
        print()

        // Output path
        let outputPath: URL
        if let customOutput = output {
            outputPath = URL(fileURLWithPath: customOutput)
        } else {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            outputPath = documentsPath.appendingPathComponent("\(triggerWord)-lora.safetensors")
        }
        print("Output: \(outputPath.path)")
        print()

        // Create training configuration
        let config = LoRATrainingConfig(
            // Dataset
            datasetPath: folderURL,
            captionExtension: "txt",
            triggerWord: triggerWord,
            imageSize: params.imageSize,
            enableBucketing: false,
            shuffleDataset: true,
            // LoRA
            rank: params.rank,
            alpha: Float(params.rank),  // alpha = rank for scale 1.0
            dropout: 0.0,
            targetLayers: .attention,
            // Training
            learningRate: params.learningRate,
            batchSize: 1,
            epochs: params.epochs,
            maxSteps: nil,
            warmupSteps: params.warmupSteps,
            lrScheduler: .cosine,
            weightDecay: 0.01,
            adamBeta1: 0.9,
            adamBeta2: 0.999,
            adamEpsilon: 1e-8,
            maxGradNorm: 1.0,
            gradientAccumulationSteps: 1,
            // Memory
            quantization: params.quantization,
            gradientCheckpointing: false,
            cacheLatents: true,
            cacheTextEmbeddings: false,
            cpuOffloadTextEncoder: false,
            mixedPrecision: true,
            // Output
            outputPath: outputPath,
            saveEveryNSteps: 0,  // No intermediate checkpoints in easy mode
            keepOnlyLastNCheckpoints: 1,
            validationPrompt: nil,
            validationEveryNSteps: 0,
            numValidationImages: 0,
            validationSeed: nil,
            // Logging
            logEveryNSteps: max(1, params.totalSteps / 10),
            evalEveryNSteps: 10,  // Sync GPU every 10 steps for faster training
            verbose: verbose,
            // Early stopping - enabled by default in easy mode
            enableEarlyStopping: true,
            earlyStoppingPatience: 5,  // Stop if no improvement for 5 epochs
            earlyStoppingMinDelta: 0.02,  // Require at least 2% improvement
            // Resume
            resumeFromCheckpoint: nil
        )

        // No confirmation needed - start training immediately

        // Enable debug if verbose
        if verbose {
            Flux2Debug.enableDebugMode()
        }

        // Create trainer
        let trainer = LoRATrainer(config: config, modelType: modelVariant)

        // Set up progress handler
        let progressHandler = EasyModeProgressHandler()
        trainer.setEventHandler(progressHandler)

        // Prepare training
        print()
        print("Loading models...")
        try await trainer.prepare()

        // Load VAE
        let vae = try await loadVAE()
        print("  VAE loaded")

        // Pre-cache latents
        print("Pre-caching images...")
        try await trainer.preCacheLatents(vae: vae)

        // Load text encoder
        print("Loading text encoder...")
        let textEncoder = try await loadTextEncoder(for: modelVariant, quantization: params.quantization)

        // Load transformer
        print("Loading transformer...")
        let transformer = try await loadTransformer(for: modelVariant, quantization: params.quantization)

        print()
        print("=".repeating(60))
        print("  Training in progress...")
        print("=".repeating(60))
        print()

        // Run training
        try await trainer.train(
            transformer: transformer,
            vae: nil,  // Using cached latents
            textEncoder: { prompt in
                try textEncoder.encode(prompt)
            }
        )

        print()
        print("=".repeating(60))
        print("  Training complete!")
        print("=".repeating(60))
        print()
        print("Your LoRA has been saved to:")
        print("  \(outputPath.path)")
        print()
        print("To use it, run:")
        print("  flux2 t2i --model \(model) --lora \"\(outputPath.path)\" \"a \(triggerWord) ...\"")
        print()
    }

    // MARK: - Caption Generation

    struct CaptionStats {
        let existing: Int
        let generated: Int
    }

    func prepareCaptoins(
        images: [URL],
        triggerWord: String,
        subjectType: SubjectType,
        useVLM: Bool
    ) async throws -> CaptionStats {
        var existing = 0
        var generated = 0

        for imageURL in images {
            let captionURL = imageURL.deletingPathExtension().appendingPathExtension("txt")

            if FileManager.default.fileExists(atPath: captionURL.path) {
                // Caption exists, check if it has trigger word
                var caption = try String(contentsOf: captionURL, encoding: .utf8)
                if !caption.contains(triggerWord) && !caption.contains("[trigger]") {
                    // Prepend trigger word
                    caption = "\(triggerWord) \(caption)"
                    try caption.write(to: captionURL, atomically: true, encoding: .utf8)
                }
                // Replace [trigger] placeholder
                if caption.contains("[trigger]") {
                    caption = caption.replacingOccurrences(of: "[trigger]", with: triggerWord)
                    try caption.write(to: captionURL, atomically: true, encoding: .utf8)
                }
                existing += 1
            } else {
                // Generate simple caption
                let caption: String
                if useVLM {
                    // TODO: Use VLM for detailed captions
                    caption = generateSimpleCaption(triggerWord: triggerWord, subjectType: subjectType)
                } else {
                    caption = generateSimpleCaption(triggerWord: triggerWord, subjectType: subjectType)
                }
                try caption.write(to: captionURL, atomically: true, encoding: .utf8)
                generated += 1
            }
        }

        return CaptionStats(existing: existing, generated: generated)
    }

    func generateSimpleCaption(triggerWord: String, subjectType: SubjectType) -> String {
        // Generate appropriate caption based on subject type
        switch subjectType {
        case .person:
            return "a photo of \(triggerWord) person"
        case .style:
            return "an image in \(triggerWord) style"
        case .object:
            return "a photo of \(triggerWord)"
        case .character:
            return "a photo of \(triggerWord) character"
        }
    }

    // MARK: - Parameter Calculation

    struct OptimalParameters {
        let rank: Int
        let epochs: Int
        let learningRate: Float
        let warmupSteps: Int
        let imageSize: Int
        let quantization: TrainingQuantization
        let totalSteps: Int
        let estimatedTime: String
    }

    func calculateOptimalParameters(
        imageCount: Int,
        intensity: TrainingIntensity,
        model: Flux2Model
    ) -> OptimalParameters {
        // Base parameters based on intensity
        // Key insight: target a FIXED total step count, not steps per image
        // This prevents overtraining on large datasets and undertraining on small ones
        let baseRank: Int
        let targetSteps: Int
        let baseLR: Float

        switch intensity {
        case .light:
            baseRank = 8
            targetSteps = 500   // Fixed total steps
            baseLR = 5e-5
        case .normal:
            baseRank = 16
            targetSteps = 1000  // Fixed total steps
            baseLR = 1e-4
        case .strong:
            baseRank = 32
            targetSteps = 1500  // Fixed total steps
            baseLR = 2e-4
        }

        // Calculate epochs to reach target steps
        // epochs = targetSteps / imageCount
        // But ensure minimum 5 epochs (small datasets need repetition)
        // and maximum 100 epochs (avoid excessive repetition)
        let rawEpochs = targetSteps / max(1, imageCount)
        let epochs = min(100, max(5, rawEpochs))

        // Actual total steps (may differ from target due to epoch bounds)
        let totalSteps = epochs * imageCount

        // Warmup: 10% of total steps, min 10, max 100
        let warmupSteps = min(100, max(10, totalSteps / 10))

        // Learning rate adjustment based on dataset size
        var learningRate = baseLR
        if imageCount < 10 {
            // Small dataset: lower LR to avoid overfitting
            learningRate *= 0.5
        } else if imageCount > 50 {
            // Large dataset: can use slightly higher LR
            learningRate *= 1.2
        }

        // Image size based on model
        let imageSize = 512  // Keep it manageable

        // Quantization based on model size
        let quantization: TrainingQuantization = .int8

        // Estimate time (rough: ~2.5s per step for Klein 4B)
        let estimatedSeconds = Int(Float(totalSteps) * 2.5)
        let estimatedTime: String
        if estimatedSeconds < 60 {
            estimatedTime = "\(estimatedSeconds)s"
        } else if estimatedSeconds < 3600 {
            estimatedTime = "\(estimatedSeconds / 60)m \(estimatedSeconds % 60)s"
        } else {
            estimatedTime = "\(estimatedSeconds / 3600)h \(estimatedSeconds % 3600 / 60)m"
        }

        return OptimalParameters(
            rank: baseRank,
            epochs: epochs,
            learningRate: learningRate,
            warmupSteps: warmupSteps,
            imageSize: imageSize,
            quantization: quantization,
            totalSteps: totalSteps,
            estimatedTime: estimatedTime
        )
    }

    // MARK: - Model Loading

    private func loadVAE() async throws -> AutoencoderKLFlux2 {
        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .vae(.standard)) else {
            throw ValidationError("VAE not found. Run: flux2 download --vae")
        }

        let vaePath = modelPath.appendingPathComponent("vae")
        let weightsPath = FileManager.default.fileExists(atPath: vaePath.path) ? vaePath : modelPath

        let vae = AutoencoderKLFlux2()
        let weights = try Flux2WeightLoader.loadWeights(from: weightsPath)
        try Flux2WeightLoader.applyVAEWeights(weights, to: vae)
        eval(vae.parameters())

        return vae
    }

    private func loadTextEncoder(
        for model: Flux2Model,
        quantization: TrainingQuantization
    ) async throws -> KleinTextEncoder {
        let mistralQuant: MistralQuantization
        switch quantization {
        case .bf16:
            mistralQuant = .bf16
        case .int8:
            mistralQuant = .mlx8bit
        case .int4, .nf4:
            mistralQuant = .mlx4bit
        }

        let variant: KleinVariant
        switch model {
        case .klein4B:
            variant = .klein4B
        case .klein9B:
            variant = .klein9B
        case .dev:
            throw ValidationError("Dev model not supported in easy mode. Use Klein 4B or 9B.")
        }

        let encoder = KleinTextEncoder(variant: variant, quantization: mistralQuant)
        try await encoder.load()

        return encoder
    }

    private func loadTransformer(
        for model: Flux2Model,
        quantization: TrainingQuantization
    ) async throws -> Flux2Transformer2DModel {
        let transformerQuant: TransformerQuantization
        switch quantization {
        case .bf16:
            transformerQuant = .bf16
        case .int8, .int4, .nf4:
            transformerQuant = .qint8
        }

        let variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: transformerQuant)

        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .transformer(variant)) else {
            throw ValidationError("Transformer not found. Run: flux2 download --model \(model.rawValue)")
        }

        let transformer = Flux2Transformer2DModel(
            config: model.transformerConfig,
            memoryOptimization: .aggressive
        )

        let weights = try Flux2WeightLoader.loadWeights(from: modelPath)
        try Flux2WeightLoader.applyTransformerWeights(weights, to: transformer)
        eval(transformer.parameters())

        return transformer
    }
}

// MARK: - Supporting Types

enum TrainingIntensity: String {
    case light
    case normal
    case strong

    var displayName: String {
        switch self {
        case .light: return "Light (quick, subtle effect)"
        case .normal: return "Normal (balanced)"
        case .strong: return "Strong (longer, stronger effect)"
        }
    }
}

enum SubjectType: String {
    case person
    case style
    case object
    case character

    var displayName: String {
        switch self {
        case .person: return "Person/Face"
        case .style: return "Art Style"
        case .object: return "Object/Thing"
        case .character: return "Character/Mascot"
        }
    }
}

// MARK: - Progress Handler

final class EasyModeProgressHandler: TrainingEventHandler, @unchecked Sendable {
    private var lastProgressUpdate: Date = Date()
    private let progressUpdateInterval: TimeInterval = 2.0  // Update every 2 seconds

    func handleEvent(_ event: TrainingEvent) {
        switch event {
        case .started:
            break
        case .epochStarted(let epoch):
            print("Epoch \(epoch)...")
        case .stepCompleted(let step, let loss):
            let now = Date()
            if now.timeIntervalSince(lastProgressUpdate) >= progressUpdateInterval {
                print("  Step \(step) - Loss: \(String(format: "%.4f", loss))")
                lastProgressUpdate = now
            }
        case .epochCompleted(let epoch, let avgLoss):
            print("Epoch \(epoch) complete - Avg Loss: \(String(format: "%.4f", avgLoss))")
        case .checkpointSaved(_, _):
            break
        case .validationCompleted(_, _):
            break
        case .validationLossComputed(let step, let trainLoss, let valLoss):
            let gap = valLoss - trainLoss
            let indicator = gap > 0.1 ? "⚠️ " : ""
            print("  \(indicator)Step \(step) - Train: \(String(format: "%.4f", trainLoss)) | Val: \(String(format: "%.4f", valLoss))")
        case .validationImageGenerated(_, _):
            break
        case .completed(let finalLoss, let totalSteps):
            print("Training finished! Final loss: \(String(format: "%.4f", finalLoss)) after \(totalSteps) steps")
        case .error(let error):
            print("Error: \(error.localizedDescription)")
        }
    }
}

// MARK: - String Extension

private extension String {
    func repeating(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
