// TrainingDataset.swift - Training dataset loading and batching
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// A single training sample
public struct TrainingSample: @unchecked Sendable {
    /// Image filename (for debugging/logging)
    public let filename: String
    
    /// Image data as MLXArray [H, W, C] in range [0, 1]
    public let image: MLXArray
    
    /// Caption text
    public let caption: String
    
    /// Original image size before resizing
    public let originalSize: (width: Int, height: Int)
}

/// A batch of training samples
public struct TrainingBatch: @unchecked Sendable {
    /// Batch of images [B, H, W, C]
    public let images: MLXArray
    
    /// Captions for each image in batch
    public let captions: [String]
    
    /// Filenames for logging
    public let filenames: [String]
    
    /// Batch size
    public var count: Int { captions.count }
}

/// Training dataset with image loading and batching
public final class TrainingDataset: @unchecked Sendable {
    
    /// Dataset configuration
    public let config: LoRATrainingConfig
    
    /// Loaded samples (filename -> caption)
    private var samples: [(filename: String, caption: String)] = []
    
    /// Dataset path
    private let datasetPath: URL
    
    /// Current epoch
    private(set) var currentEpoch: Int = 0
    
    /// Current index within epoch
    private var currentIndex: Int = 0
    
    /// Sample order (for shuffling)
    private var sampleOrder: [Int] = []
    
    /// Initialize training dataset
    /// - Parameter config: Training configuration
    public init(config: LoRATrainingConfig) throws {
        self.config = config
        self.datasetPath = config.datasetPath
        
        // Parse captions
        let parser = CaptionParser(triggerWord: config.triggerWord)
        self.samples = try parser.parseDataset(
            at: datasetPath,
            extension: config.captionExtension
        )
        
        guard !samples.isEmpty else {
            throw TrainingDatasetError.emptyDataset
        }
        
        Flux2Debug.log("[TrainingDataset] Loaded \(samples.count) samples")
        
        // Initialize sample order
        resetOrder()
    }
    
    // MARK: - Dataset Info
    
    /// Number of samples in dataset
    public var count: Int { samples.count }
    
    /// Number of batches per epoch
    public var batchesPerEpoch: Int {
        (samples.count + config.batchSize - 1) / config.batchSize
    }
    
    /// Total number of training steps
    public var totalSteps: Int {
        if let maxSteps = config.maxSteps {
            return maxSteps
        }
        return batchesPerEpoch * config.epochs
    }
    
    // MARK: - Iteration
    
    /// Reset sample order (optionally shuffle)
    private func resetOrder() {
        sampleOrder = Array(0..<samples.count)
        if config.shuffleDataset {
            sampleOrder.shuffle()
        }
        currentIndex = 0
    }
    
    /// Start a new epoch
    public func startEpoch() {
        currentEpoch += 1
        resetOrder()
        Flux2Debug.log("[TrainingDataset] Starting epoch \(currentEpoch)")
    }
    
    /// Get next batch of samples
    /// - Returns: Training batch, or nil if epoch is complete
    public func nextBatch() throws -> TrainingBatch? {
        guard currentIndex < samples.count else {
            return nil
        }
        
        let endIndex = Swift.min(currentIndex + config.batchSize, samples.count)
        let batchIndices = sampleOrder[currentIndex..<endIndex]
        currentIndex = endIndex
        
        var images: [MLXArray] = []
        var captions: [String] = []
        var filenames: [String] = []
        
        for idx in batchIndices {
            let sample = samples[idx]
            let imagePath = datasetPath.appendingPathComponent(sample.filename)
            
            // Load and preprocess image
            let image = try loadImage(at: imagePath, targetSize: config.imageSize)
            
            images.append(image)
            captions.append(sample.caption)
            filenames.append(sample.filename)
        }
        
        // Stack images into batch
        let batchedImages = MLX.stacked(images, axis: 0)
        
        return TrainingBatch(
            images: batchedImages,
            captions: captions,
            filenames: filenames
        )
    }
    
    /// Get a specific sample
    public func getSample(at index: Int) throws -> TrainingSample {
        guard index >= 0 && index < samples.count else {
            throw TrainingDatasetError.indexOutOfBounds(index)
        }
        
        let sample = samples[index]
        let imagePath = datasetPath.appendingPathComponent(sample.filename)
        let image = try loadImage(at: imagePath, targetSize: config.imageSize)
        
        return TrainingSample(
            filename: sample.filename,
            image: image,
            caption: sample.caption,
            originalSize: (config.imageSize, config.imageSize)
        )
    }
    
    // MARK: - Image Loading
    
    /// Load and preprocess an image
    /// - Parameters:
    ///   - url: Path to the image file
    ///   - targetSize: Target size for resizing
    /// - Returns: MLXArray of shape [H, W, C] in range [0, 1]
    private func loadImage(at url: URL, targetSize: Int) throws -> MLXArray {
        // Load image using ImageIO
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw TrainingDatasetError.failedToLoadImage(url)
        }
        
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw TrainingDatasetError.failedToLoadImage(url)
        }
        
        // Get original dimensions
        let originalWidth = cgImage.width
        let originalHeight = cgImage.height
        
        // Create bitmap context for resizing and conversion
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = targetSize * 4
        
        guard let context = CGContext(
            data: nil,
            width: targetSize,
            height: targetSize,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw TrainingDatasetError.failedToCreateContext
        }
        
        // Calculate crop/resize to fit target size (center crop)
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
        
        let aspectRatio = CGFloat(originalWidth) / CGFloat(originalHeight)
        let targetAspect: CGFloat = 1.0  // Square for now
        
        if aspectRatio > targetAspect {
            // Image is wider than target - scale by height, crop width
            scale = CGFloat(targetSize) / CGFloat(originalHeight)
            offsetX = (CGFloat(originalWidth) * scale - CGFloat(targetSize)) / 2
            offsetY = 0
        } else {
            // Image is taller than target - scale by width, crop height
            scale = CGFloat(targetSize) / CGFloat(originalWidth)
            offsetX = 0
            offsetY = (CGFloat(originalHeight) * scale - CGFloat(targetSize)) / 2
        }
        
        let drawRect = CGRect(
            x: -offsetX,
            y: -offsetY,
            width: CGFloat(originalWidth) * scale,
            height: CGFloat(originalHeight) * scale
        )
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: drawRect)
        
        // Get pixel data
        guard let pixelData = context.data else {
            throw TrainingDatasetError.failedToGetPixelData
        }
        
        // Convert to MLXArray [H, W, C] normalized to [0, 1]
        let pixels = pixelData.bindMemory(to: UInt8.self, capacity: targetSize * targetSize * 4)
        var floatPixels: [Float] = []
        floatPixels.reserveCapacity(targetSize * targetSize * 3)
        
        for i in 0..<(targetSize * targetSize) {
            let r = Float(pixels[i * 4]) / 255.0
            let g = Float(pixels[i * 4 + 1]) / 255.0
            let b = Float(pixels[i * 4 + 2]) / 255.0
            floatPixels.append(r)
            floatPixels.append(g)
            floatPixels.append(b)
        }
        
        let array = MLXArray(floatPixels, [targetSize, targetSize, 3])
        
        return array
    }
    
    // MARK: - Validation
    
    /// Validate the dataset
    public func validate() -> DatasetValidationResult {
        let parser = CaptionParser(triggerWord: config.triggerWord)
        return parser.validateDataset(at: datasetPath, extension: config.captionExtension)
    }
    
    /// Get sample statistics
    public func getStatistics() -> DatasetStatistics {
        let captionLengths = samples.map { $0.caption.count }
        
        return DatasetStatistics(
            totalSamples: samples.count,
            minCaptionLength: captionLengths.min() ?? 0,
            maxCaptionLength: captionLengths.max() ?? 0,
            avgCaptionLength: captionLengths.isEmpty ? 0 : captionLengths.reduce(0, +) / captionLengths.count
        )
    }
}

// MARK: - Supporting Types

/// Dataset statistics
public struct DatasetStatistics: Sendable {
    public let totalSamples: Int
    public let minCaptionLength: Int
    public let maxCaptionLength: Int
    public let avgCaptionLength: Int
    
    public var summary: String {
        """
        Dataset Statistics:
          Total samples: \(totalSamples)
          Caption length: min=\(minCaptionLength), max=\(maxCaptionLength), avg=\(avgCaptionLength)
        """
    }
}

/// Dataset errors
public enum TrainingDatasetError: Error, LocalizedError {
    case emptyDataset
    case indexOutOfBounds(Int)
    case failedToLoadImage(URL)
    case failedToCreateContext
    case failedToGetPixelData
    case invalidImageFormat(URL)
    
    public var errorDescription: String? {
        switch self {
        case .emptyDataset:
            return "Dataset contains no valid samples"
        case .indexOutOfBounds(let index):
            return "Sample index \(index) out of bounds"
        case .failedToLoadImage(let url):
            return "Failed to load image: \(url.lastPathComponent)"
        case .failedToCreateContext:
            return "Failed to create image processing context"
        case .failedToGetPixelData:
            return "Failed to extract pixel data from image"
        case .invalidImageFormat(let url):
            return "Invalid image format: \(url.lastPathComponent)"
        }
    }
}

// MARK: - Iterator Protocol

extension TrainingDataset: Sequence {
    public struct Iterator: IteratorProtocol {
        private let dataset: TrainingDataset
        private var index: Int = 0
        
        init(dataset: TrainingDataset) {
            self.dataset = dataset
        }
        
        public mutating func next() -> TrainingSample? {
            guard index < dataset.count else { return nil }
            let sample = try? dataset.getSample(at: index)
            index += 1
            return sample
        }
    }
    
    public func makeIterator() -> Iterator {
        Iterator(dataset: self)
    }
}
