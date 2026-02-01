// CheckpointManager.swift - Save and load training checkpoints
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXOptimizers

/// Manages saving and loading of training checkpoints
public final class CheckpointManager: @unchecked Sendable {
    
    /// Output directory for checkpoints
    public let outputDirectory: URL
    
    /// Maximum number of checkpoints to keep
    public let maxCheckpoints: Int
    
    /// List of saved checkpoint paths
    private var savedCheckpoints: [URL] = []
    
    /// Initialize checkpoint manager
    /// - Parameters:
    ///   - outputDirectory: Directory to save checkpoints
    ///   - maxCheckpoints: Maximum number of checkpoints to keep (0 = unlimited)
    public init(outputDirectory: URL, maxCheckpoints: Int = 3) {
        self.outputDirectory = outputDirectory
        self.maxCheckpoints = maxCheckpoints
        
        // Create output directory if needed
        try? FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        
        // Scan for existing checkpoints
        scanExistingCheckpoints()
    }
    
    // MARK: - Save Checkpoint
    
    /// Save a training checkpoint
    /// - Parameters:
    ///   - injector: LoRA injector with trainable parameters
    ///   - state: Current training state
    ///   - config: Training configuration
    /// - Returns: Path to saved checkpoint
    @discardableResult
    public func saveCheckpoint(
        injector: LoRAInjector,
        state: TrainingState,
        config: LoRATrainingConfig
    ) throws -> URL {
        let checkpointDir = outputDirectory
            .appendingPathComponent("checkpoint-\(state.globalStep)")
        
        // Create checkpoint directory
        try FileManager.default.createDirectory(
            at: checkpointDir,
            withIntermediateDirectories: true
        )
        
        // Save LoRA weights
        let loraPath = checkpointDir.appendingPathComponent("lora.safetensors")
        try injector.saveWeights(to: loraPath)
        
        // Save training state
        let statePath = checkpointDir.appendingPathComponent("state.json")
        let stateData = try JSONEncoder().encode(state)
        try stateData.write(to: statePath)
        
        // Save config for reference
        let configPath = checkpointDir.appendingPathComponent("config.json")
        try config.save(to: configPath)
        
        // Save metadata
        let metadata = CheckpointMetadata(
            step: state.globalStep,
            epoch: state.epoch,
            loss: state.currentLoss,
            timestamp: Date(),
            rank: config.rank,
            alpha: config.alpha
        )
        let metadataPath = checkpointDir.appendingPathComponent("metadata.json")
        let metadataData = try JSONEncoder().encode(metadata)
        try metadataData.write(to: metadataPath)
        
        // Track saved checkpoint
        savedCheckpoints.append(checkpointDir)
        
        // Clean up old checkpoints
        cleanupOldCheckpoints()
        
        Flux2Debug.log("[CheckpointManager] Saved checkpoint at step \(state.globalStep)")
        
        return checkpointDir
    }
    
    /// Save final LoRA weights (without state)
    public func saveFinalWeights(
        injector: LoRAInjector,
        config: LoRATrainingConfig,
        outputPath: URL
    ) throws {
        // Save just the LoRA weights in standard format
        try injector.saveWeights(to: outputPath)
        
        // Also save metadata alongside
        let metadataPath = outputPath.deletingPathExtension()
            .appendingPathExtension("json")
        
        let metadata = LoRAWeightsMetadata(
            rank: config.rank,
            alpha: config.alpha,
            targetLayers: config.targetLayers.rawValue,
            triggerWord: config.triggerWord,
            trainedOn: Date()
        )
        
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: metadataPath)
        
        Flux2Debug.log("[CheckpointManager] Saved final weights to \(outputPath.path)")
    }
    
    // MARK: - Load Checkpoint
    
    /// Load a training checkpoint
    /// - Parameters:
    ///   - checkpointDir: Path to checkpoint directory
    ///   - injector: LoRA injector to load weights into
    /// - Returns: Loaded training state
    public func loadCheckpoint(
        from checkpointDir: URL,
        into injector: LoRAInjector
    ) throws -> TrainingState {
        // Load LoRA weights
        let loraPath = checkpointDir.appendingPathComponent("lora.safetensors")
        try injector.loadWeights(from: loraPath)
        
        // Load training state
        let statePath = checkpointDir.appendingPathComponent("state.json")
        let stateData = try Data(contentsOf: statePath)
        let state = try JSONDecoder().decode(TrainingState.self, from: stateData)
        
        Flux2Debug.log("[CheckpointManager] Loaded checkpoint from step \(state.globalStep)")
        
        return state
    }
    
    /// Get the latest checkpoint directory
    public func getLatestCheckpoint() -> URL? {
        savedCheckpoints.last
    }
    
    /// Get checkpoint for a specific step
    public func getCheckpoint(forStep step: Int) -> URL? {
        let checkpointDir = outputDirectory.appendingPathComponent("checkpoint-\(step)")
        return FileManager.default.fileExists(atPath: checkpointDir.path) ? checkpointDir : nil
    }
    
    // MARK: - Checkpoint Management
    
    /// Scan for existing checkpoints in output directory
    private func scanExistingCheckpoints() {
        let fileManager = FileManager.default
        
        guard let contents = try? fileManager.contentsOfDirectory(
            at: outputDirectory,
            includingPropertiesForKeys: nil
        ) else { return }
        
        // Find checkpoint directories
        let checkpoints = contents.filter {
            $0.lastPathComponent.hasPrefix("checkpoint-")
        }.sorted { url1, url2 in
            // Sort by step number
            let step1 = extractStep(from: url1)
            let step2 = extractStep(from: url2)
            return step1 < step2
        }
        
        savedCheckpoints = checkpoints
    }
    
    /// Extract step number from checkpoint path
    private func extractStep(from url: URL) -> Int {
        let name = url.lastPathComponent
        let stepStr = name.replacingOccurrences(of: "checkpoint-", with: "")
        return Int(stepStr) ?? 0
    }
    
    /// Clean up old checkpoints to maintain maxCheckpoints limit
    private func cleanupOldCheckpoints() {
        guard maxCheckpoints > 0, savedCheckpoints.count > maxCheckpoints else { return }
        
        // Remove oldest checkpoints
        let toRemove = savedCheckpoints.count - maxCheckpoints
        let checkpointsToDelete = Array(savedCheckpoints.prefix(toRemove))
        
        for checkpoint in checkpointsToDelete {
            do {
                try FileManager.default.removeItem(at: checkpoint)
                savedCheckpoints.removeAll { $0 == checkpoint }
                Flux2Debug.log("[CheckpointManager] Removed old checkpoint: \(checkpoint.lastPathComponent)")
            } catch {
                Flux2Debug.log("[CheckpointManager] Failed to remove checkpoint: \(error)")
            }
        }
    }
    
    /// List all available checkpoints
    public func listCheckpoints() -> [CheckpointInfo] {
        savedCheckpoints.compactMap { dir in
            let metadataPath = dir.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: metadataPath),
                  let metadata = try? JSONDecoder().decode(CheckpointMetadata.self, from: data) else {
                return nil
            }
            
            return CheckpointInfo(
                path: dir,
                step: metadata.step,
                epoch: metadata.epoch,
                loss: metadata.loss,
                timestamp: metadata.timestamp
            )
        }
    }
}

// MARK: - Supporting Types

/// Metadata stored with each checkpoint
public struct CheckpointMetadata: Codable, Sendable {
    public let step: Int
    public let epoch: Int
    public let loss: Float
    public let timestamp: Date
    public let rank: Int
    public let alpha: Float
}

/// Metadata stored with final LoRA weights
public struct LoRAWeightsMetadata: Codable, Sendable {
    public let rank: Int
    public let alpha: Float
    public let targetLayers: String
    public let triggerWord: String?
    public let trainedOn: Date
    /// Whether EMA weights were used (if available)
    public let usedEMA: Bool?
    /// EMA decay factor used (if EMA was enabled)
    public let emaDecay: Float?

    public init(
        rank: Int,
        alpha: Float,
        targetLayers: String,
        triggerWord: String?,
        trainedOn: Date,
        usedEMA: Bool? = nil,
        emaDecay: Float? = nil
    ) {
        self.rank = rank
        self.alpha = alpha
        self.targetLayers = targetLayers
        self.triggerWord = triggerWord
        self.trainedOn = trainedOn
        self.usedEMA = usedEMA
        self.emaDecay = emaDecay
    }
}

/// Information about a saved checkpoint
public struct CheckpointInfo: Sendable {
    public let path: URL
    public let step: Int
    public let epoch: Int
    public let loss: Float
    public let timestamp: Date
    
    public var summary: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        
        return "Step \(step) | Epoch \(epoch) | Loss: \(String(format: "%.4f", loss)) | \(formatter.string(from: timestamp))"
    }
}

// MARK: - Errors

public enum CheckpointError: Error, LocalizedError {
    case checkpointNotFound(URL)
    case invalidCheckpointFormat(URL)
    case loadFailed(String)
    case saveFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .checkpointNotFound(let url):
            return "Checkpoint not found: \(url.path)"
        case .invalidCheckpointFormat(let url):
            return "Invalid checkpoint format: \(url.path)"
        case .loadFailed(let message):
            return "Failed to load checkpoint: \(message)"
        case .saveFailed(let message):
            return "Failed to save checkpoint: \(message)"
        }
    }
}
