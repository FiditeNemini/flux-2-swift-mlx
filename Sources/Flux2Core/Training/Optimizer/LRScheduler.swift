// LRScheduler.swift - Learning rate schedulers for training
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Protocol for learning rate schedulers
public protocol LearningRateScheduler: Sendable {
    /// Get learning rate for current step
    func getLearningRate(step: Int) -> Float
    
    /// Total number of steps
    var totalSteps: Int { get }
    
    /// Number of warmup steps
    var warmupSteps: Int { get }
    
    /// Base learning rate
    var baseLR: Float { get }
}

/// Constant learning rate with optional warmup
public struct ConstantScheduler: LearningRateScheduler {
    public let baseLR: Float
    public let warmupSteps: Int
    public let totalSteps: Int
    
    public init(baseLR: Float, warmupSteps: Int = 0, totalSteps: Int = 1000) {
        self.baseLR = baseLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
    }
    
    public func getLearningRate(step: Int) -> Float {
        if step < warmupSteps {
            return baseLR * Float(step + 1) / Float(warmupSteps)
        }
        return baseLR
    }
}

/// Linear learning rate decay with optional warmup
public struct LinearScheduler: LearningRateScheduler {
    public let baseLR: Float
    public let warmupSteps: Int
    public let totalSteps: Int
    public let endLR: Float
    
    public init(
        baseLR: Float,
        warmupSteps: Int = 0,
        totalSteps: Int = 1000,
        endLR: Float = 0.0
    ) {
        self.baseLR = baseLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.endLR = endLR
    }
    
    public func getLearningRate(step: Int) -> Float {
        // Warmup phase
        if step < warmupSteps {
            return baseLR * Float(step + 1) / Float(warmupSteps)
        }
        
        // Decay phase
        let decaySteps = totalSteps - warmupSteps
        let currentDecayStep = step - warmupSteps
        let progress = Float(currentDecayStep) / Float(max(decaySteps, 1))
        
        return baseLR + (endLR - baseLR) * progress
    }
}

/// Cosine annealing learning rate scheduler with optional warmup
public struct CosineScheduler: LearningRateScheduler {
    public let baseLR: Float
    public let warmupSteps: Int
    public let totalSteps: Int
    public let minLR: Float
    
    public init(
        baseLR: Float,
        warmupSteps: Int = 0,
        totalSteps: Int = 1000,
        minLR: Float = 0.0
    ) {
        self.baseLR = baseLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.minLR = minLR
    }
    
    public func getLearningRate(step: Int) -> Float {
        // Warmup phase
        if step < warmupSteps {
            return baseLR * Float(step + 1) / Float(warmupSteps)
        }
        
        // Cosine decay phase
        let decaySteps = totalSteps - warmupSteps
        let currentDecayStep = step - warmupSteps
        let progress = Float(currentDecayStep) / Float(max(decaySteps, 1))
        
        // Cosine annealing: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        let cosineDecay = 0.5 * (1.0 + cos(Float.pi * progress))
        return minLR + (baseLR - minLR) * cosineDecay
    }
}

/// Cosine annealing with warm restarts
public struct CosineWithRestartsScheduler: LearningRateScheduler {
    public let baseLR: Float
    public let warmupSteps: Int
    public let totalSteps: Int
    public let minLR: Float
    public let numCycles: Int
    
    public init(
        baseLR: Float,
        warmupSteps: Int = 0,
        totalSteps: Int = 1000,
        minLR: Float = 0.0,
        numCycles: Int = 1
    ) {
        self.baseLR = baseLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.minLR = minLR
        self.numCycles = numCycles
    }
    
    public func getLearningRate(step: Int) -> Float {
        // Warmup phase
        if step < warmupSteps {
            return baseLR * Float(step + 1) / Float(warmupSteps)
        }
        
        // Cosine with restarts
        let decaySteps = totalSteps - warmupSteps
        let currentDecayStep = step - warmupSteps
        
        // Progress within current cycle
        let cycleLength = Float(decaySteps) / Float(numCycles)
        let currentCycleProgress = Float(currentDecayStep).truncatingRemainder(dividingBy: cycleLength)
        let progress = currentCycleProgress / cycleLength
        
        // Cosine annealing within cycle
        let cosineDecay = 0.5 * (1.0 + cos(Float.pi * progress))
        return minLR + (baseLR - minLR) * cosineDecay
    }
}

/// Polynomial learning rate decay
public struct PolynomialScheduler: LearningRateScheduler {
    public let baseLR: Float
    public let warmupSteps: Int
    public let totalSteps: Int
    public let endLR: Float
    public let power: Float
    
    public init(
        baseLR: Float,
        warmupSteps: Int = 0,
        totalSteps: Int = 1000,
        endLR: Float = 0.0,
        power: Float = 2.0
    ) {
        self.baseLR = baseLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.endLR = endLR
        self.power = power
    }
    
    public func getLearningRate(step: Int) -> Float {
        // Warmup phase
        if step < warmupSteps {
            return baseLR * Float(step + 1) / Float(warmupSteps)
        }
        
        // Polynomial decay
        let decaySteps = totalSteps - warmupSteps
        let currentDecayStep = step - warmupSteps
        let progress = Float(currentDecayStep) / Float(max(decaySteps, 1))
        
        let decay = pow(1.0 - progress, power)
        return endLR + (baseLR - endLR) * decay
    }
}

// MARK: - Factory

/// Factory for creating learning rate schedulers
public struct LRSchedulerFactory {
    
    /// Create scheduler from configuration
    public static func create(
        type: LRSchedulerType,
        baseLR: Float,
        warmupSteps: Int,
        totalSteps: Int,
        options: LRSchedulerOptions = .default
    ) -> LearningRateScheduler {
        switch type {
        case .constant:
            return ConstantScheduler(
                baseLR: baseLR,
                warmupSteps: warmupSteps,
                totalSteps: totalSteps
            )
            
        case .linear:
            return LinearScheduler(
                baseLR: baseLR,
                warmupSteps: warmupSteps,
                totalSteps: totalSteps,
                endLR: options.endLR
            )
            
        case .cosine:
            return CosineScheduler(
                baseLR: baseLR,
                warmupSteps: warmupSteps,
                totalSteps: totalSteps,
                minLR: options.minLR
            )
            
        case .cosineWithRestarts:
            return CosineWithRestartsScheduler(
                baseLR: baseLR,
                warmupSteps: warmupSteps,
                totalSteps: totalSteps,
                minLR: options.minLR,
                numCycles: options.numCycles
            )
        }
    }
}

/// Options for learning rate schedulers
public struct LRSchedulerOptions: Sendable {
    /// End learning rate (for linear decay)
    public var endLR: Float
    
    /// Minimum learning rate (for cosine)
    public var minLR: Float
    
    /// Number of cycles (for cosine with restarts)
    public var numCycles: Int
    
    /// Power (for polynomial decay)
    public var power: Float
    
    public init(
        endLR: Float = 0.0,
        minLR: Float = 0.0,
        numCycles: Int = 1,
        power: Float = 2.0
    ) {
        self.endLR = endLR
        self.minLR = minLR
        self.numCycles = numCycles
        self.power = power
    }
    
    public static let `default` = LRSchedulerOptions()
}

// MARK: - Warmup Helper

/// Linear warmup wrapper for any scheduler
public struct WarmupWrapper: LearningRateScheduler {
    private let inner: LearningRateScheduler
    public let warmupSteps: Int
    
    public var baseLR: Float { inner.baseLR }
    public var totalSteps: Int { inner.totalSteps }
    
    public init(wrapping scheduler: LearningRateScheduler, warmupSteps: Int) {
        self.inner = scheduler
        self.warmupSteps = warmupSteps
    }
    
    public func getLearningRate(step: Int) -> Float {
        if step < warmupSteps {
            return baseLR * Float(step + 1) / Float(warmupSteps)
        }
        return inner.getLearningRate(step: step)
    }
}
