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

    public func configuration(id: String) -> ModelConfiguration {
        lock.withLock {
            if let c = registry[id] {
                return c
            } else {
                return ModelConfiguration(id: id)
            }
        }
    }

    public var models: some Collection<ModelConfiguration> & Sendable {
        lock.withLock {
            return registry.values
        }
    }
}
