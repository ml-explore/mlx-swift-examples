// Copyright © 2024 Apple Inc.

import Foundation
import Tokenizers

open class ProcessorTypeRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor])
    {
        self.creators = creators
    }

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    private var creators: [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor]

    /// Add a new model to the type registry.
    public func registerProcessorType(
        _ type: String,
        creator: @Sendable @escaping (
            URL,
            any Tokenizer
        ) throws -> any UserInputProcessor
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
