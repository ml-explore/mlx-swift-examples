// Copyright © 2024 Apple Inc.

import Foundation

open class ModelTypeRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: @Sendable (URL) throws -> any LanguageModel]) {
        self.creators = creators
    }

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel]

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any LanguageModel
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    /// Given a `modelType` and configuration file instantiate a new `LanguageModel`.
    public func createModel(configuration: URL, modelType: String) throws -> LanguageModel {
        let creator = lock.withLock {
            creators[modelType]
        }
        guard let creator else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

}
