// Copyright © 2024 Apple Inc.

import Foundation
@preconcurrency import Hub
import MLX
import MLXNN
import Tokenizers

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer.  Callers can access
/// the model and/or tokenizer:
///
/// ```swift
/// let promptTokens = await modelContainer.perform { _, tokenizer in
///     tokenizer.encode(text: prompt)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let result = await modelContainer.perform { model, tokenizer in
///     LLM.generate(
///         promptTokens: promptTokens, parameters: generateParameters, model: model,
///         tokenizer: tokenizer, extraEOSTokens: modelConfiguration.extraEOSTokens
///     ) { tokens in
///     ...
///     }
/// }
/// ```
public actor ModelContainer {
    let model: LLMModel
    let tokenizer: Tokenizer

    public init(model: LLMModel, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public init(model: LLMModel, configuration: ModelConfiguration, hub: HubApi) async throws {
        self.model = model

        let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
            configuration: configuration, hub: hub)

        self.tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    /// build the model and tokenizer without passing non-sendable data over isolation barriers
    public init(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws {
        let modelDirectory = try await prepareModelDirectory(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
        self.model = try loadSynchronous(modelDirectory: modelDirectory)

        let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
            configuration: configuration, hub: hub)
        self.tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    /// Perform an action on the model and/or tokenizer.  Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<R>(_ action: @Sendable (LLMModel, Tokenizer) throws -> R) rethrows -> R {
        try action(model, tokenizer)
    }
}

/// Interface for all LLM Models
public protocol LLMModel: Module {

    var vocabularySize: Int { get }

    func callAsFunction(_ inputs: MLXArray, cache: [(MLXArray, MLXArray)]?) -> (
        MLXArray, [(MLXArray, MLXArray)]
    )

    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

extension LLMModel {

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

}
