//
//  ModelAdapterTypeRegistry.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 06.06.2025.
//

import Foundation

public class ModelAdapterTypeRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: @Sendable (URL) throws -> any ModelAdapter]) {
        self.creators = creators
    }

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()
    private var creators: [String: @Sendable (URL) throws -> any ModelAdapter]

    /// Add a new model adapter to the type registry.
    public func registerAdapterType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any ModelAdapter
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    public func createAdapter(directory: URL, adapterType: String) throws -> ModelAdapter {
        let creator = lock.withLock {
            creators[adapterType]
        }
        guard let creator else {
            throw ModelAdapterError.unsupportedAdapterType(adapterType)
        }
        return try creator(directory)
    }
}
