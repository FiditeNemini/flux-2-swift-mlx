// Training.swift - LoRA Training Module for Flux.2
// Copyright 2025 Vincent Gourbin

// This file serves as the module entry point for training functionality.
// All public types from the Training subdirectories are re-exported here.

// MARK: - Configuration
// LoRATrainingConfig - Main configuration for training
// TrainingQuantization - Quantization options (bf16, int8, int4, nf4)
// LoRATargetLayers - Which layers to apply LoRA to
// LRSchedulerType - Learning rate scheduler types

// MARK: - Data Pipeline
// TrainingDataset - Load and iterate over training data
// TrainingSample - Single training sample
// TrainingBatch - Batch of training samples
// CaptionParser - Parse captions from txt/jsonl files
// LatentCache - Pre-encode and cache VAE latents
// TextEmbeddingCache - Cache text encoder outputs

// MARK: - Model
// TrainableLoRALinear - LoRA layer with gradient support
// LoRAInjector - Inject LoRA into transformer

// MARK: - Optimizer & Loss
// DiffusionLoss - MSE loss for diffusion training
// TrainingStepLoss - Complete loss computation
// LearningRateScheduler - Protocol for LR schedulers
// ConstantScheduler, LinearScheduler, CosineScheduler - LR scheduler implementations

// MARK: - Training Loop
// LoRATrainer - Main training loop
// TrainingState - Track training progress
// CheckpointManager - Save/load checkpoints
// TrainingEvent - Events during training
// TrainingEventHandler - Handle training events

import Foundation

/// LoRA Training Module Version
public enum Training {
    public static let version = "0.1.0"
    
    /// Supported features
    public static let features: [String] = [
        "LoRA training for Flux.2 models",
        "Support for Klein 4B, Klein 9B, and Dev models",
        "Multiple quantization options (bf16, int8, int4, nf4)",
        "Gradient checkpointing for memory efficiency",
        "Latent and text embedding caching",
        "Multiple learning rate schedulers",
        "Checkpoint saving and resumption",
        "Compatible with Kohya-SS/SimpleTuner dataset formats"
    ]
    
    /// Minimum recommended memory by model and quantization
    public static func recommendedMemoryGB(
        for model: Flux2Model,
        quantization: TrainingQuantization
    ) -> Int {
        switch (model, quantization) {
        case (.klein4B, .nf4): return 8
        case (.klein4B, .int4): return 8
        case (.klein4B, .int8): return 12
        case (.klein4B, .bf16): return 16
        case (.klein9B, .nf4): return 12
        case (.klein9B, .int4): return 12
        case (.klein9B, .int8): return 16
        case (.klein9B, .bf16): return 24
        case (.dev, .nf4): return 18
        case (.dev, .int4): return 18
        case (.dev, .int8): return 24
        case (.dev, .bf16): return 48
        }
    }
}
