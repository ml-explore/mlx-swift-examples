// Copyright © 2024 Apple Inc.

import AsyncAlgorithms
import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers

struct LLMError: Error {
    let message: String
}

/// Load and return the model and tokenizer
public func load(
    hub: HubApi = HubApi(), configuration: ModelConfiguration,
    progressHandler: @escaping (Progress) -> Void = { _ in }
) async throws -> (LLMModel, Tokenizer) {
    do {
        let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

        let modelDirectory: URL

        switch configuration.id {
        case .id(let id):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors"]
            modelDirectory = try await hub.snapshot(
                from: repo, matching: modelFiles, progressHandler: progressHandler)

        case .directory(let directory):
            modelDirectory = directory
        }

        // create the model (no weights loaded)
        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))

        let model = try baseConfig.modelType.createModel(configuration: configurationURL)

        // load the weights
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

        // per-model cleanup
        weights = model.sanitize(weights: weights)

        // quantize if needed
        if let quantization = baseConfig.quantization {
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) {
                path, module in
                weights["\(path).scales"] != nil
            }
        }

        // apply the loaded weights
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])

        eval(model)

        return (model, tokenizer)

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        var newConfiguration = configuration
        newConfiguration.id = .directory(configuration.modelDirectory(hub: hub))
        return try await load(
            hub: hub, configuration: newConfiguration, progressHandler: progressHandler)
    }
}
