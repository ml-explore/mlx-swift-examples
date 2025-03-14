// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json"]
            return try await hub.snapshot(
                from: repo, matching: modelFiles, progressHandler: progressHandler)

        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel, quantization: BaseConfiguration.Quantization? = nil
) throws {
    // Load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }
    // Per-model cleanup
    weights = model.sanitize(weights: weights)
    // Apply quantization if needed
    if let quantization = quantization {
        // Check if we should skip quantizing vision components
        let skipVision = shouldSkipVision(modelDirectory: modelDirectory)
        let predicate = getClassPredicate(skipVision: skipVision, weights: weights)
        // Use the Swift quantize function with our predicate
        quantize(
            model: model,
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            filter: predicate
        )
    }
    // Apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
}

// Creates a predicate for determining which modules to quantize
private func getClassPredicate(skipVision: Bool, weights: [String: MLXArray]? = nil) -> (
    String, Module
) -> Bool {
    if skipVision {
        // Don't quantize vision components
        return { path, module in
            // Check if module is quantizable (has to_quantized method in Python)
            let isQuantizable = module is Quantizable
            // Don't quantize vision components
            let isNotVisionComponent =
                !path.contains("vision_model") && !path.contains("vision_tower")
            return isQuantizable && isNotVisionComponent
        }
    } else {
        if let weights = weights {
            // Only quantize modules that have scales in the weights
            return { path, module in
                // Check if module is quantizable
                let isQuantizable = module is Quantizable
                // Check if module has appropriate dimensions (weight.shape[-1] % 64 == 0 in Python)
                var hasDivisibleDimensions = false
                if let linear = module as? Linear, let lastShapeElement = linear.weight.shape.last {
                    hasDivisibleDimensions = lastShapeElement % 64 == 0
                } else if let embedding = module as? Embedding,
                    let lastShapeElement = embedding.weight.shape.last
                {
                    hasDivisibleDimensions = lastShapeElement % 64 == 0
                }
                // Check if scales exist in weights
                let hasScales = weights["\(path).scales"] != nil
                return isQuantizable && hasDivisibleDimensions && hasScales
            }
        } else {
            // Default case - quantize modules with appropriate dimensions
            return { _, module in
                // Check if module is quantizable
                let isQuantizable = module is Quantizable
                // Check if module has appropriate dimensions
                var hasDivisibleDimensions = false
                if let linear = module as? Linear, let lastShapeElement = linear.weight.shape.last {
                    hasDivisibleDimensions = lastShapeElement % 64 == 0
                } else if let embedding = module as? Embedding,
                    let lastShapeElement = embedding.weight.shape.last
                {
                    hasDivisibleDimensions = lastShapeElement % 64 == 0
                }
                return isQuantizable && hasDivisibleDimensions
            }
        }
    }
}

// Helper function to check if vision components should be skipped during quantization
private func shouldSkipVision(modelDirectory: URL) -> Bool {
    let configURL = modelDirectory.appending(component: "config.json")
    guard let configData = try? Data(contentsOf: configURL),
        let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else {
        return false
    }
    // Check if this is a model with vision components that should be skipped
    if let visionConfig = json["vision_config"] as? [String: Any],
        let skipVision = visionConfig["skip_vision"] as? Bool
    {
        return skipVision
    }
    return false
}
