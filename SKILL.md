---
name: flux2-swift-mlx
description: Use when generating images with Flux.2 on Apple Silicon, working with MLX Swift, or implementing text-to-image/image-to-image pipelines in Swift.
---

# Flux.2 Swift MLX

Native Swift implementation of Flux.2 image generation for Apple Silicon.

## Quick Start

```swift
import Flux2Core

// Create pipeline (Klein 4B for fast generation)
let pipeline = Flux2Pipeline(model: .klein4B)
try await pipeline.loadModels()

// Generate image
let image = try await pipeline.generateTextToImage(
    prompt: "a cat sitting on a chair",
    height: 1024,
    width: 1024,
    steps: 4,        // Klein uses 4 steps
    guidance: 1.0    // Klein uses guidance 1.0
)
```

## Model Selection

| Model | Steps | Guidance | RAM | Speed | License |
|-------|-------|----------|-----|-------|---------|
| Klein 4B | 4 | 1.0 | 16GB+ | ~26s | Apache 2.0 |
| Klein 9B | 4 | 1.0 | 32GB+ | ~62s | Non-commercial |
| Dev (32B) | 28 | 4.0 | 64GB+ | ~35min | Non-commercial |

## Common Patterns

### Text-to-Image with Result

```swift
// Get both image and the prompt that was actually used
let result = try await pipeline.generateTextToImageWithResult(
    prompt: "a sunset",
    upsamplePrompt: true  // Enhance with Mistral/Qwen3
)

print("Used prompt: \(result.usedPrompt)")
let image = result.image
```

### Image-to-Image

```swift
// Transform existing images
let image = try await pipeline.generateImageToImage(
    prompt: "transform into watercolor style",
    images: [referenceImage],
    strength: 0.7  // 0.0 = preserve original, 1.0 = full transform
)
```

### Multi-Image Conditioning

```swift
// Combine elements from multiple images
let image = try await pipeline.generateImageToImage(
    prompt: "a cat wearing this hat",
    images: [catImage, hatImage],  // Up to 3 images
    steps: 28,
    guidance: 4.0
)
```

## Memory Management

The pipeline uses two-phase loading:
1. Text encoder loads first (then unloads)
2. Transformer + VAE load for generation

This allows running on machines with less RAM than the total model size.

```swift
// Clear memory when done
await pipeline.clearAll()
```

## CLI Usage

```bash
# Text-to-Image
flux2 t2i "a beautiful landscape" --model klein-4b --output image.png

# Image-to-Image
flux2 i2i "make it look like a painting" --images photo.jpg --strength 0.7
```
