// TrainingState.swift - Track training progress and state
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Training progress state
public struct TrainingState: Codable, Sendable {
    
    // MARK: - Progress Tracking
    
    /// Current global step
    public var globalStep: Int
    
    /// Current epoch
    public var epoch: Int
    
    /// Step within current epoch
    public var epochStep: Int
    
    /// Total samples processed
    public var samplesProcessed: Int
    
    /// Total number of steps planned
    public var totalSteps: Int
    
    /// Total number of epochs planned
    public var totalEpochs: Int
    
    // MARK: - Loss Tracking
    
    /// Current loss value
    public var currentLoss: Float
    
    /// Best loss seen
    public var bestLoss: Float
    
    /// Loss history (last N values for averaging)
    public var lossHistory: [Float]
    
    /// Step at which best loss was achieved
    public var bestLossStep: Int
    
    // MARK: - Timing
    
    /// Training start time
    public var startTime: Date
    
    /// Last checkpoint time
    public var lastCheckpointTime: Date?
    
    /// Last validation time
    public var lastValidationTime: Date?
    
    /// Total training duration in seconds
    public var trainingDuration: TimeInterval {
        Date().timeIntervalSince(startTime)
    }
    
    // MARK: - Initialization
    
    public init(
        totalSteps: Int,
        totalEpochs: Int
    ) {
        self.globalStep = 0
        self.epoch = 0
        self.epochStep = 0
        self.samplesProcessed = 0
        self.totalSteps = totalSteps
        self.totalEpochs = totalEpochs
        self.currentLoss = 0
        self.bestLoss = Float.infinity
        self.lossHistory = []
        self.bestLossStep = 0
        self.startTime = Date()
        self.lastCheckpointTime = nil
        self.lastValidationTime = nil
    }
    
    // MARK: - Progress Updates
    
    /// Update state after a training step
    public mutating func update(
        loss: Float,
        batchSize: Int
    ) {
        globalStep += 1
        epochStep += 1
        samplesProcessed += batchSize

        // Only update loss tracking if we have a valid loss value
        // (NaN is used when eval is skipped for performance)
        guard !loss.isNaN else { return }

        currentLoss = loss

        // Update loss history (keep last 100)
        lossHistory.append(loss)
        if lossHistory.count > 100 {
            lossHistory.removeFirst()
        }

        // Track best loss
        if loss < bestLoss {
            bestLoss = loss
            bestLossStep = globalStep
        }
    }
    
    /// Start a new epoch
    public mutating func startEpoch() {
        epoch += 1
        epochStep = 0
    }
    
    // MARK: - Computed Properties
    
    /// Progress as fraction [0, 1]
    public var progress: Float {
        Float(globalStep) / Float(max(totalSteps, 1))
    }
    
    /// Progress as percentage
    public var progressPercent: Int {
        Int(progress * 100)
    }
    
    /// Average loss over recent history
    public var averageLoss: Float {
        guard !lossHistory.isEmpty else { return 0 }
        return lossHistory.reduce(0, +) / Float(lossHistory.count)
    }
    
    /// Estimated time remaining in seconds
    public var estimatedTimeRemaining: TimeInterval {
        guard globalStep > 0 else { return 0 }
        let timePerStep = trainingDuration / Double(globalStep)
        let remainingSteps = totalSteps - globalStep
        return timePerStep * Double(remainingSteps)
    }
    
    /// Steps per second
    public var stepsPerSecond: Float {
        guard trainingDuration > 0 else { return 0 }
        return Float(globalStep) / Float(trainingDuration)
    }
    
    /// Samples per second
    public var samplesPerSecond: Float {
        guard trainingDuration > 0 else { return 0 }
        return Float(samplesProcessed) / Float(trainingDuration)
    }
    
    /// Whether training is complete
    public var isComplete: Bool {
        globalStep >= totalSteps
    }
    
    // MARK: - Formatting
    
    /// Format time interval as string
    private func formatDuration(_ interval: TimeInterval) -> String {
        let hours = Int(interval) / 3600
        let minutes = (Int(interval) % 3600) / 60
        let seconds = Int(interval) % 60
        
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        } else {
            return String(format: "%d:%02d", minutes, seconds)
        }
    }
    
    /// Progress summary string
    public var progressSummary: String {
        """
        Step \(globalStep)/\(totalSteps) (\(progressPercent)%) | \
        Epoch \(epoch)/\(totalEpochs) | \
        Loss: \(String(format: "%.4f", currentLoss)) (avg: \(String(format: "%.4f", averageLoss))) | \
        Time: \(formatDuration(trainingDuration)) | \
        ETA: \(formatDuration(estimatedTimeRemaining))
        """
    }
    
    /// Detailed status string
    public var detailedStatus: String {
        """
        Training Progress:
          Step: \(globalStep)/\(totalSteps) (\(progressPercent)%)
          Epoch: \(epoch)/\(totalEpochs), Step in epoch: \(epochStep)
          Samples processed: \(samplesProcessed)
        
        Loss:
          Current: \(String(format: "%.6f", currentLoss))
          Average (last 100): \(String(format: "%.6f", averageLoss))
          Best: \(String(format: "%.6f", bestLoss)) (step \(bestLossStep))
        
        Timing:
          Elapsed: \(formatDuration(trainingDuration))
          Remaining: \(formatDuration(estimatedTimeRemaining))
          Speed: \(String(format: "%.2f", stepsPerSecond)) steps/s, \(String(format: "%.2f", samplesPerSecond)) samples/s
        """
    }
}

// MARK: - Training Events

/// Events that can occur during training
public enum TrainingEvent: Sendable {
    /// Training started
    case started
    
    /// Epoch started
    case epochStarted(epoch: Int)
    
    /// Step completed
    case stepCompleted(step: Int, loss: Float)
    
    /// Checkpoint saved
    case checkpointSaved(path: String, step: Int)
    
    /// Validation completed
    case validationCompleted(loss: Float, step: Int)
    
    /// Validation image generated
    case validationImageGenerated(path: String, step: Int)
    
    /// Epoch completed
    case epochCompleted(epoch: Int, avgLoss: Float)
    
    /// Training completed
    case completed(finalLoss: Float, totalSteps: Int)
    
    /// Error occurred
    case error(Error)
}

/// Protocol for receiving training events
public protocol TrainingEventHandler: AnyObject, Sendable {
    func handleEvent(_ event: TrainingEvent)
}

/// Default event handler that prints to console
public final class ConsoleTrainingEventHandler: TrainingEventHandler, @unchecked Sendable {
    
    public init() {}
    
    public func handleEvent(_ event: TrainingEvent) {
        switch event {
        case .started:
            print("[Training] Started")
            
        case .epochStarted(let epoch):
            print("[Training] Epoch \(epoch) started")
            
        case .stepCompleted(let step, let loss):
            if step % 10 == 0 {
                print("[Training] Step \(step) | Loss: \(String(format: "%.4f", loss))")
            }
            
        case .checkpointSaved(let path, let step):
            print("[Training] Checkpoint saved at step \(step): \(path)")
            
        case .validationCompleted(let loss, let step):
            print("[Training] Validation at step \(step) | Loss: \(String(format: "%.4f", loss))")
            
        case .validationImageGenerated(let path, let step):
            print("[Training] Generated validation image at step \(step): \(path)")
            
        case .epochCompleted(let epoch, let avgLoss):
            print("[Training] Epoch \(epoch) completed | Avg Loss: \(String(format: "%.4f", avgLoss))")
            
        case .completed(let finalLoss, let totalSteps):
            print("[Training] Completed! Final loss: \(String(format: "%.4f", finalLoss)) after \(totalSteps) steps")
            
        case .error(let error):
            print("[Training] Error: \(error.localizedDescription)")
        }
    }
}

// MARK: - Progress Callback

/// Callback type for training progress updates
public typealias TrainingProgressCallback = @Sendable (TrainingState) -> Void
