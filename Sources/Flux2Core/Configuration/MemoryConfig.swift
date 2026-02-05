// MemoryConfig.swift - Centralized memory management configuration
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Centralized memory configuration for GPU cache management
public struct MemoryConfig {

    // MARK: - Cache Profiles

    /// Memory profile presets based on available RAM
    public enum CacheProfile: String, CaseIterable, Sendable {
        case conservative  // 512 MB - For 16-32 GB RAM
        case balanced      // 1.5 GB - For 32-64 GB RAM
        case performance   // 3 GB - For 64-128 GB RAM
        case unlimited     // No limit - For 128+ GB RAM

        /// Cache limit in bytes, nil means no limit
        public var cacheLimitBytes: Int? {
            switch self {
            case .conservative: return 512 * 1024 * 1024       // 512 MB
            case .balanced: return 1536 * 1024 * 1024          // 1.5 GB
            case .performance: return 3 * 1024 * 1024 * 1024   // 3 GB
            case .unlimited: return nil
            }
        }

        /// Human-readable description
        public var description: String {
            switch self {
            case .conservative: return "Conservative (512 MB) - Best for 16-32 GB RAM"
            case .balanced: return "Balanced (1.5 GB) - Best for 32-64 GB RAM"
            case .performance: return "Performance (3 GB) - Best for 64-128 GB RAM"
            case .unlimited: return "Unlimited - Best for 128+ GB RAM"
            }
        }
    }

    // MARK: - Auto-Detection

    /// Auto-detect recommended profile based on system RAM
    public static func recommendedProfile() -> CacheProfile {
        let ramGB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        switch ramGB {
        case ..<32: return .conservative
        case 32..<64: return .balanced
        case 64..<128: return .performance
        default: return .unlimited
        }
    }

    /// Get system RAM in GB
    public static var systemRAMGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    // MARK: - Cache Limit Application

    /// Apply cache limit for current profile
    public static func applyCacheLimit(_ profile: CacheProfile) {
        if let limit = profile.cacheLimitBytes {
            Memory.cacheLimit = limit
            Flux2Debug.log("GPU cache limit set to \(limit / (1024 * 1024)) MB (\(profile.rawValue) profile)")
        } else {
            Flux2Debug.log("GPU cache limit: unlimited (\(profile.rawValue) profile)")
        }
    }

    /// Apply cache limit with specific byte value
    public static func applyCacheLimit(bytes: Int) {
        Memory.cacheLimit = bytes
        Flux2Debug.log("GPU cache limit set to \(bytes / (1024 * 1024)) MB")
    }

    /// Clear GPU cache
    public static func clearCache() {
        Memory.clearCache()
    }

    // MARK: - Phase-Specific Limits

    /// Per-phase cache limits for more granular control
    public struct PhaseLimits: Sendable {
        public let textEncoding: Int   // Text encoder phase
        public let denoising: Int      // Transformer denoising loop
        public let vaeDecoding: Int    // VAE decode phase

        public init(textEncoding: Int, denoising: Int, vaeDecoding: Int) {
            self.textEncoding = textEncoding
            self.denoising = denoising
            self.vaeDecoding = vaeDecoding
        }

        /// Get recommended phase limits for a model and profile
        public static func forModel(_ model: Flux2Model, profile: CacheProfile) -> PhaseLimits {
            let MB = 1024 * 1024
            let GB = 1024 * 1024 * 1024

            switch (model, profile) {
            // Dev model (large Mistral encoder + large transformer)
            case (.dev, .conservative):
                return PhaseLimits(textEncoding: 512 * MB, denoising: 1 * GB, vaeDecoding: 512 * MB)
            case (.dev, .balanced):
                return PhaseLimits(textEncoding: 1 * GB, denoising: 2 * GB, vaeDecoding: 1 * GB)
            case (.dev, .performance), (.dev, .unlimited):
                return PhaseLimits(textEncoding: 2 * GB, denoising: 3 * GB, vaeDecoding: 2 * GB)

            // Klein 4B (smaller, more memory efficient)
            case (.klein4B, .conservative):
                return PhaseLimits(textEncoding: 256 * MB, denoising: 512 * MB, vaeDecoding: 256 * MB)
            case (.klein4B, .balanced):
                return PhaseLimits(textEncoding: 512 * MB, denoising: 1 * GB, vaeDecoding: 512 * MB)
            case (.klein4B, .performance), (.klein4B, .unlimited):
                return PhaseLimits(textEncoding: 1 * GB, denoising: 2 * GB, vaeDecoding: 1 * GB)

            // Klein 9B (medium size)
            case (.klein9B, .conservative):
                return PhaseLimits(textEncoding: 512 * MB, denoising: 1 * GB, vaeDecoding: 512 * MB)
            case (.klein9B, .balanced):
                return PhaseLimits(textEncoding: 1 * GB, denoising: 2 * GB, vaeDecoding: 1 * GB)
            case (.klein9B, .performance), (.klein9B, .unlimited):
                return PhaseLimits(textEncoding: 2 * GB, denoising: 3 * GB, vaeDecoding: 2 * GB)
            }
        }
    }

    // MARK: - Memory Monitoring

    /// Log current memory state using existing memory manager
    public static func logMemoryState(context: String = "") {
        let manager = Flux2MemoryManager.shared
        let prefix = context.isEmpty ? "" : "[\(context)] "
        Flux2Debug.log("\(prefix)System RAM: \(manager.physicalMemoryGB) GB, estimated available: ~\(manager.estimatedAvailableMemoryGB) GB")
    }
}
