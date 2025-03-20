// Copyright © 2024 Apple Inc.

import Foundation

open class AbstractModelRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    public init() {
        self.registry = Dictionary()
    }

    /// Creates a new registry with from given model configurations.
    public init(modelConfigurations: [ModelConfiguration]) {
        self.registry = Dictionary(uniqueKeysWithValues: modelConfigurations.map { ($0.name, $0) })
    }

    private let lock = NSLock()
    private var registry: [String: ModelConfiguration]

    public func register(configurations: [ModelConfiguration]) {
        lock.withLock {
            for c in configurations {
                registry[c.name] = c
            }
        }
    }

    /// Returns configuration from ``modelRegistry``.
    ///
    /// - Note: If the id doesn't exists in the configuration, this will return a new instance of it.
    /// If you want to check if the configuration in model registry, you should use ``contains(id:)``.
    public func configuration(id: String) -> ModelConfiguration {
        lock.withLock {
            if let c = registry[id] {
                return c
            } else {
                return ModelConfiguration(id: id)
            }
        }
    }

    /// Returns true if the registry contains a model with the id. Otherwise, false.
    public func contains(id: String) -> Bool {
        lock.withLock {
            registry[id] != nil
        }
    }

    public var models: some Collection<ModelConfiguration> & Sendable {
        lock.withLock {
            return registry.values
        }
    }
}
