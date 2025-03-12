// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public enum VLMError: LocalizedError {
    case imageRequired
    case maskRequired
    case singleImageAllowed
    case imageProcessingFailure(String)
    case processing(String)

    public var errorDescription: String? {
        switch self {
        case .imageRequired:
            return String(localized: "An image is required for this operation.")
        case .maskRequired:
            return String(localized: "An image mask is required for this operation.")
        case .singleImageAllowed:
            return String(localized: "Only a single image is allowed for this operation.")
        case .imageProcessingFailure(let details):
            return String(localized: "Failed to process the image: \(details)")
        case .processing(let details):
            return String(localized: "Processing error: \(details)")
        }
    }
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
    _ configurationType: C.Type,
    _ processorInit: @escaping (
        C,
        any Tokenizer
    ) -> P
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
public class VLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {

    /// Shared instance with default model types.
    public static let shared: VLMTypeRegistry = .init(creators: all())

    /// All predefined model types
    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
        [
            "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
            "qwen2_vl": create(Qwen2VLConfiguration.self, Qwen2VL.init),
            "idefics3": create(Idefics3Configuration.self, Idefics3.init),
        ]
    }

}

public class VLMProcessorTypeRegistry: ProcessorTypeRegistry, @unchecked Sendable {

    /// Shared instance with default processor types.
    public static let shared: VLMProcessorTypeRegistry = .init(creators: all())

    /// All predefined processor types.
    private static func all() -> [String: @Sendable (URL, any Tokenizer) throws ->
        any UserInputProcessor]
    {
        [
            "PaliGemmaProcessor": create(
                PaliGemmaProcessorConfiguration.self, PaligGemmaProcessor.init),
            "Qwen2VLProcessor": create(Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
            "Idefics3Processor": create(
                Idefics3ProcessorConfiguration.self, Idefics3Processor.init),
        ]
    }

}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class VLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared: VLMRegistry = .init(modelConfigurations: all())

    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2VL2BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let smolvlminstruct4bit = ModelConfiguration(
        id: "mlx-community/SmolVLM-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static private func all() -> [ModelConfiguration] {
        [
            paligemma3bMix448_8bit,
            qwen2VL2BInstruct4Bit,
            smolvlminstruct4bit,
        ]
    }

}

@available(*, deprecated, renamed: "VLMRegistry", message: "Please use VLMRegistry directly.")
public typealias ModelRegistry = VLMRegistry

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

    public init(
        typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.processorRegistry = processorRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = VLMModelFactory(
        typeRegistry: VLMTypeRegistry.shared, processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)

    /// registry of model type, e.g. configuration value `paligemma` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of input processor type, e.g. configuration value `PaliGemmaProcessor` -> configuration and init methods
    public let processorRegistry: ProcessorTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/paligemma-3b-mix-448-8bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler)

        // load the generic config to unerstand which model and how to load the weights
        let configurationURL = modelDirectory.appending(
            component: "config.json"
        )
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))

        let model = try typeRegistry.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization)

        let tokenizer = try await loadTokenizer(
            configuration: configuration,
            hub: hub
        )

        let processorConfiguration = modelDirectory.appending(
            component: "preprocessor_config.json"
        )
        let baseProcessorConfig = try JSONDecoder().decode(
            BaseProcessorConfiguration.self,
            from: Data(
                contentsOf: processorConfiguration
            )
        )
        let processor = try processorRegistry.createModel(
            configuration: processorConfiguration,
            processorType: baseProcessorConfig.processorClass, tokenizer: tokenizer)

        return .init(
            configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)
    }

}
