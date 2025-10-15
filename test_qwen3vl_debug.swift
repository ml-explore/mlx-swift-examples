#!/usr/bin/env swift

// Minimal test to check if embeddings are loading correctly
// Usage: swift test_qwen3vl_debug.swift

import Foundation
import MLX
import MLXVLM

let modelPath = "mlx-community/Qwen3-VL-4B-Instruct-8bit"

print("Loading model...")
let (model, processor) = try! await VLMModelFactory.shared.loadContainer(
    configuration: .init(id: .id(modelPath))
).get()

// Test with simple token IDs
let testTokens = MLXArray([151644, 8948, 198, 2610])  // Some tokens from vocab
print("Test tokens: \(testTokens)")

// Get embeddings
if let qwen3vl = model.wrappedValue as? Qwen3VL {
    let embeds = qwen3vl.languageModel.model.embedTokens(testTokens)
    print("Embeddings shape: \(embeds.shape)")
    print("Embeddings mean: \(embeds.mean().item(Float.self))")
    print("Embeddings std: \(embeds.variance().sqrt().item(Float.self))")
    print("First 5 values of first token: \(embeds[0, 0..<5].asArray(Float.self))")
} else {
    print("ERROR: Model is not Qwen3VL type")
}
