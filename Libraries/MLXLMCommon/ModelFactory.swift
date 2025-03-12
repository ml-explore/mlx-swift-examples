// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public enum ModelFactoryError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type): "Unsupported model type: \(type)"
        case .unsupportedProcessorType(let type): "Unsupported processor type: \(type)"
        }
    }
}

/// Context of types that work together to provide a ``LanguageModel``.
///
/// A ``ModelContext`` is created by ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This contains the following:
///
/// - ``ModelConfiguration`` -- identifier for the model
/// - ``LanguageModel`` -- the model itself, see ``generate(input:parameters:context:didGenerate:)``
/// - ``UserInputProcessor`` -- can convert ``UserInput`` into ``LMInput``
/// - `Tokenizer` -- the tokenizer used by ``UserInputProcessor``
///
/// See also ``ModelFactory/loadContainer(hub:configuration:progressHandler:)`` and
/// ``ModelContainer``.
public struct ModelContext {
    public var configuration: ModelConfiguration
    public var model: any LanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer

    public init(
        configuration: ModelConfiguration, model: any LanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        self.configuration = configuration
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    }
}

public protocol ModelFactory: Sendable {

    var modelRegistry: AbstractModelRegistry { get }

    func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext

    func _loadContainer(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContainer

}

extension ModelFactory {

    /// Resolve a model identifier, e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit", into
    /// a ``ModelConfiguration``.
    ///
    /// This will either create a new (mostly unconfigured) ``ModelConfiguration`` or
    /// return a registered instance that matches the id.
    ///
    /// - Note: If the id doesn't exists in the configuration, this will return a new instance of it.
    /// If you want to check if the configuration in model registry, you should use ``contains(id:)``.
    public func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    /// Returns true if ``modelRegistry`` contains a model with the id. Otherwise, false.
    public func contains(id: String) -> Bool {
        modelRegistry.contains(id: id)
    }

}

extension ModelFactory {

    /// Load a model identified by a ``ModelConfiguration`` and produce a ``ModelContext``.
    ///
    /// This method returns a ``ModelContext``.  See also
    /// ``loadContainer(hub:configuration:progressHandler:)`` for a method that
    /// returns a ``ModelContainer``.
    public func load(
        hub: HubApi = HubApi(), configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        try await _load(hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    /// Load a model identified by a ``ModelConfiguration`` and produce a ``ModelContainer``.
    public func loadContainer(
        hub: HubApi = HubApi(), configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        try await _loadContainer(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    public func _loadContainer(
        hub: HubApi = HubApi(), configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await _load(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
        return ModelContainer(context: context)
    }

}
