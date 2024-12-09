// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public enum VLMError: Error {
    case imageRequired
    case maskRequired
    case singleImageAllowed
    case imageProcessingFailure(String)
}

public struct BaseProcessorConfiguration: Codable, Sendable {
    public let processorClass: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
    }
}

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (URL) throws -> M {
    { url in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url))
        return modelInit(configuration)
    }
}

private func create<C: Codable, P>(
    _ configurationType: C.Type, _ processorInit: @escaping (C, any Tokenizer) -> P
) -> (URL, any Tokenizer) throws -> P {
    { url, tokenizer in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url))
        return processorInit(configuration, tokenizer)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(hub:configuration:progressHandler:)``.
public class ModelTypeRegistry: @unchecked Sendable {

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
        "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
        "qwen2_vl": create(Qwen2VLConfiguration.self, Qwen2VL.init),
    ]

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any LanguageModel
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    /// Given a `modelType` and configuration file instantiate a new `LanguageModel`.
    public func createModel(configuration: URL, modelType: String) throws -> any LanguageModel {
        let creator = lock.withLock {
            creators[modelType]
        }
        guard let creator else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

}

public class ProcessorTypeRegistry: @unchecked Sendable {

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    private var creators:
        [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor] = [
            "PaliGemmaProcessor": create(
                PaliGemmaProcessorConfiguration.self, PaligGemmaProcessor.init),
            "Qwen2VLProcessor": create(
                Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
        ]

    /// Add a new model to the type registry.
    public func registerProcessorType(
        _ type: String,
        creator: @Sendable @escaping (URL, any Tokenizer) throws -> any UserInputProcessor
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    /// Given a `processorType` and configuration file instantiate a new `UserInputProcessor`.
    public func createModel(configuration: URL, processorType: String, tokenizer: any Tokenizer)
        throws -> any UserInputProcessor
    {
        let creator = lock.withLock {
            creators[processorType]
        }
        guard let creator else {
            throw ModelFactoryError.unsupportedProcessorType(processorType)
        }
        return try creator(configuration, tokenizer)
    }

}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class ModelRegistry: @unchecked Sendable {

    private let lock = NSLock()
    private var registry = Dictionary(uniqueKeysWithValues: all().map { ($0.name, $0) })

    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2VL2BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static private func all() -> [ModelConfiguration] {
        [
            paligemma3bMix448_8bit,
            qwen2VL2BInstruct4Bit,
        ]
    }

    public func register(configurations: [ModelConfiguration]) {
        lock.withLock {
            for c in configurations {
                registry[c.name] = c
            }
        }
    }

    public func configuration(id: String) -> ModelConfiguration {
        lock.withLock {
            if let c = registry[id] {
                return c
            } else {
                return ModelConfiguration(id: id)
            }
        }
    }
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await VLMModelFactory.shared.loadContainer(
///     configuration: ModelRegistry.paligemma3bMix4488bit)
/// ```
public class VLMModelFactory: ModelFactory {

    public static let shared = VLMModelFactory()

    /// registry of model type, e.g. configuration value `paligemma` -> configuration and init methods
    public let typeRegistry = ModelTypeRegistry()

    /// registry of input processor type, e.g. configuration value `PaliGemmaProcessor` -> configuration and init methods
    public let processorRegistry = ProcessorTypeRegistry()

    /// registry of model id to configuration, e.g. `mlx-community/paligemma-3b-mix-448-8bit`
    public let modelRegistry = ModelRegistry()

    public func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler)

        // load the generic config to unerstand which model and how to load the weights
        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))

        let model = try typeRegistry.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization)

        let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

        let processorConfiguration = modelDirectory.appending(component: "preprocessor_config.json")
        let baseProcessorConfig = try JSONDecoder().decode(
            BaseProcessorConfiguration.self, from: Data(contentsOf: processorConfiguration))
        let processor = try processorRegistry.createModel(
            configuration: processorConfiguration,
            processorType: baseProcessorConfig.processorClass, tokenizer: tokenizer)

        return .init(
            configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)
    }

}
