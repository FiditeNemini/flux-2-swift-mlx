// LatentCache.swift - Pre-compute and cache VAE latents for training
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Cached latent entry
public struct CachedLatent: @unchecked Sendable {
    /// Original filename
    public let filename: String
    
    /// Caption text
    public let caption: String
    
    /// Latent representation [H/8, W/8, C]
    public let latent: MLXArray
    
    /// Image size used for encoding
    public let imageSize: Int
}

/// Cache for pre-encoded VAE latents
///
/// Pre-encoding images with VAE and caching the latents provides:
/// - ~50% memory savings during training (no need to keep VAE in memory)
/// - Faster training (no VAE encoding per step)
/// - Consistent latents across epochs
public final class LatentCache: @unchecked Sendable {
    
    /// Cache directory path
    public let cacheDirectory: URL
    
    /// Configuration
    public let config: LoRATrainingConfig
    
    /// In-memory cache (optional)
    private var memoryCache: [String: MLXArray] = [:]
    
    /// Whether to keep latents in memory
    public let useMemoryCache: Bool
    
    /// Initialize latent cache
    /// - Parameters:
    ///   - config: Training configuration
    ///   - cacheDirectory: Directory to store cached latents
    ///   - useMemoryCache: Whether to also cache in memory
    public init(
        config: LoRATrainingConfig,
        cacheDirectory: URL? = nil,
        useMemoryCache: Bool = true
    ) {
        self.config = config
        self.useMemoryCache = useMemoryCache
        
        // Default cache directory next to dataset
        if let dir = cacheDirectory {
            self.cacheDirectory = dir
        } else {
            self.cacheDirectory = config.datasetPath
                .appendingPathComponent(".latent_cache")
        }
        
        // Create cache directory if needed
        try? FileManager.default.createDirectory(
            at: self.cacheDirectory,
            withIntermediateDirectories: true
        )
    }
    
    // MARK: - Cache Management
    
    /// Check if latent is cached
    public func isCached(filename: String) -> Bool {
        if useMemoryCache && memoryCache[filename] != nil {
            return true
        }
        
        let cacheFile = cacheFilePath(for: filename)
        return FileManager.default.fileExists(atPath: cacheFile.path)
    }
    
    /// Get cache file path for a filename
    private func cacheFilePath(for filename: String) -> URL {
        let baseName = (filename as NSString).deletingPathExtension
        return cacheDirectory.appendingPathComponent("\(baseName)_latent.safetensors")
    }
    
    /// Get cached latent
    public func getLatent(for filename: String) throws -> MLXArray? {
        // Check memory cache first
        if useMemoryCache, let latent = memoryCache[filename] {
            return latent
        }
        
        // Load from disk
        let cacheFile = cacheFilePath(for: filename)
        guard FileManager.default.fileExists(atPath: cacheFile.path) else {
            return nil
        }
        
        let weights = try loadArrays(url: cacheFile)
        guard let latent = weights["latent"] else {
            return nil
        }
        
        // Optionally store in memory
        if useMemoryCache {
            memoryCache[filename] = latent
        }
        
        return latent
    }
    
    /// Save latent to cache
    public func saveLatent(_ latent: MLXArray, for filename: String) throws {
        // Save to memory
        if useMemoryCache {
            memoryCache[filename] = latent
        }
        
        // Save to disk
        let cacheFile = cacheFilePath(for: filename)
        try save(arrays: ["latent": latent], url: cacheFile)
    }
    
    /// Clear all cached latents
    public func clearCache() throws {
        memoryCache.removeAll()
        
        let contents = try FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        )
        
        for file in contents where file.pathExtension == "safetensors" {
            try FileManager.default.removeItem(at: file)
        }
        
        Flux2Debug.log("[LatentCache] Cleared all cached latents")
    }
    
    /// Clear memory cache only (keep disk cache)
    public func clearMemoryCache() {
        memoryCache.removeAll()
        Flux2Debug.log("[LatentCache] Cleared memory cache")
    }
    
    // MARK: - Pre-encoding
    
    /// Pre-encode all images in dataset and cache latents
    /// - Parameters:
    ///   - dataset: Training dataset
    ///   - vae: VAE encoder
    ///   - progressCallback: Called with (current, total) progress
    /// - Returns: Number of latents cached
    @discardableResult
    public func preEncodeDataset(
        _ dataset: TrainingDataset,
        vae: AutoencoderKLFlux2,
        progressCallback: ((Int, Int) -> Void)? = nil
    ) async throws -> Int {
        var encodedCount = 0
        let total = dataset.count
        
        Flux2Debug.log("[LatentCache] Pre-encoding \(total) images...")
        
        for (index, sample) in dataset.enumerated() {
            // Check if already cached
            if isCached(filename: sample.filename) {
                encodedCount += 1
                progressCallback?(index + 1, total)
                continue
            }
            
            // Encode with VAE
            // Image should be in range [0, 1], need to convert to [-1, 1]
            let normalizedImage = sample.image * 2.0 - 1.0

            // Add batch dimension [1, H, W, C] then transpose to [1, C, H, W] (NCHW)
            let batchedImage = normalizedImage.expandedDimensions(axis: 0)
            let nchwImage = batchedImage.transposed(0, 3, 1, 2)  // NHWC -> NCHW

            // Encode to latent space
            let latent = vae.encode(nchwImage)
            
            // Remove batch dimension for storage
            let squeezedLatent = latent.squeezed(axis: 0)
            
            // Save to cache
            try saveLatent(squeezedLatent, for: sample.filename)
            
            encodedCount += 1
            progressCallback?(index + 1, total)
            
            // Periodically clear GPU cache
            if index % 10 == 0 {
                eval(squeezedLatent)
                MLX.Memory.clearCache()
            }
        }
        
        Flux2Debug.log("[LatentCache] Pre-encoded \(encodedCount) latents")
        
        return encodedCount
    }
    
    /// Get latent for a batch (loading from cache or encoding)
    /// - Parameters:
    ///   - batch: Training batch
    ///   - vae: VAE encoder (used if not cached)
    /// - Returns: Batched latents [B, H/8, W/8, C]
    public func getLatents(
        for batch: TrainingBatch,
        vae: AutoencoderKLFlux2?
    ) throws -> MLXArray {
        var latents: [MLXArray] = []
        
        for (i, filename) in batch.filenames.enumerated() {
            if let cached = try getLatent(for: filename) {
                latents.append(cached)
            } else if let vae = vae {
                // Encode on the fly
                let image = batch.images[i]
                let normalizedImage = image * 2.0 - 1.0
                let batchedImage = normalizedImage.expandedDimensions(axis: 0)
                let nchwImage = batchedImage.transposed(0, 3, 1, 2)  // NHWC -> NCHW
                let latent = vae.encode(nchwImage).squeezed(axis: 0)
                
                // Cache for next time
                try saveLatent(latent, for: filename)
                latents.append(latent)
            } else {
                throw LatentCacheError.latentNotCached(filename)
            }
        }
        
        return MLX.stacked(latents, axis: 0)
    }
    
    // MARK: - Statistics
    
    /// Get cache statistics
    public func getStatistics() -> CacheStatistics {
        let fileManager = FileManager.default
        
        var diskCount = 0
        var diskSize: Int64 = 0
        
        if let files = try? fileManager.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: [.fileSizeKey]
        ) {
            for file in files where file.pathExtension == "safetensors" {
                diskCount += 1
                if let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    diskSize += Int64(size)
                }
            }
        }
        
        return CacheStatistics(
            memoryCacheCount: memoryCache.count,
            diskCacheCount: diskCount,
            diskCacheSizeMB: Float(diskSize) / (1024 * 1024)
        )
    }
}

// MARK: - Supporting Types

/// Cache statistics
public struct CacheStatistics: Sendable {
    public let memoryCacheCount: Int
    public let diskCacheCount: Int
    public let diskCacheSizeMB: Float
    
    public var summary: String {
        """
        Latent Cache Statistics:
          Memory cache: \(memoryCacheCount) entries
          Disk cache: \(diskCacheCount) files (\(String(format: "%.1f", diskCacheSizeMB)) MB)
        """
    }
}

/// Cache errors
public enum LatentCacheError: Error, LocalizedError {
    case latentNotCached(String)
    case failedToSave(String)
    case failedToLoad(String)
    
    public var errorDescription: String? {
        switch self {
        case .latentNotCached(let filename):
            return "Latent not cached for: \(filename)"
        case .failedToSave(let filename):
            return "Failed to save latent for: \(filename)"
        case .failedToLoad(let filename):
            return "Failed to load latent for: \(filename)"
        }
    }
}

// MARK: - Text Embedding Cache

/// Cache for pre-computed text embeddings
public final class TextEmbeddingCache: @unchecked Sendable {
    
    /// Cache directory path
    public let cacheDirectory: URL
    
    /// In-memory cache
    private var memoryCache: [String: (pooled: MLXArray, hidden: MLXArray)] = [:]
    
    /// Initialize text embedding cache
    public init(cacheDirectory: URL) {
        self.cacheDirectory = cacheDirectory
        
        // Create cache directory
        try? FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
    }
    
    /// Check if embedding is cached
    public func isCached(caption: String) -> Bool {
        let key = cacheKey(for: caption)
        if memoryCache[key] != nil { return true }
        
        let cacheFile = cacheDirectory.appendingPathComponent("\(key).safetensors")
        return FileManager.default.fileExists(atPath: cacheFile.path)
    }
    
    /// Get cache key for caption
    private func cacheKey(for caption: String) -> String {
        // Use hash of caption as key
        let hash = caption.hashValue
        return String(format: "emb_%016llx", UInt64(bitPattern: Int64(hash)))
    }
    
    /// Get cached embeddings
    public func getEmbeddings(for caption: String) throws -> (pooled: MLXArray, hidden: MLXArray)? {
        let key = cacheKey(for: caption)
        
        // Check memory cache
        if let cached = memoryCache[key] {
            return cached
        }
        
        // Load from disk
        let cacheFile = cacheDirectory.appendingPathComponent("\(key).safetensors")
        guard FileManager.default.fileExists(atPath: cacheFile.path) else {
            return nil
        }
        
        let weights = try loadArrays(url: cacheFile)
        guard let pooled = weights["pooled"],
              let hidden = weights["hidden"] else {
            return nil
        }
        
        let result = (pooled: pooled, hidden: hidden)
        memoryCache[key] = result
        return result
    }
    
    /// Save embeddings to cache
    public func saveEmbeddings(
        pooled: MLXArray,
        hidden: MLXArray,
        for caption: String
    ) throws {
        let key = cacheKey(for: caption)
        memoryCache[key] = (pooled: pooled, hidden: hidden)
        
        let cacheFile = cacheDirectory.appendingPathComponent("\(key).safetensors")
        try save(arrays: ["pooled": pooled, "hidden": hidden], url: cacheFile)
    }
    
    /// Clear cache
    public func clearCache() throws {
        memoryCache.removeAll()
        
        let contents = try FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        )
        
        for file in contents where file.pathExtension == "safetensors" {
            try FileManager.default.removeItem(at: file)
        }
    }
}
