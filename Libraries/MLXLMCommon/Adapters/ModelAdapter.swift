//
//  ModelAdapters.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN
import Hub

/// Errors that can occur when working with a `ModelAdapter`.
public enum ModelAdapterError: Error {
    case incompatibleModelType
}

/// Protocol defining an adapter that can modify a `LanguageModel`.
public protocol ModelAdapter: Sendable {
    
    /// Loads the adapter into the specified model.
    func load(into model: LanguageModel) throws
    
    /// Unloads the adapter from the specified model.
    func unload(from model: LanguageModel)
    
    /// Permanently fuses the adapter into the specified model.
    func fuse(with model: LanguageModel) throws
}

/// Extension to `LanguageModel` providing convenience methods for adapter usage.
public extension LanguageModel {
    
    /// Loads an adapter into the model.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try model.load(adapter: adapter)
    /// ```
    func load(adapter: ModelAdapter) throws {
        try adapter.load(into: self)
    }
    
    /// Unloads an adapter from the model.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// model.unload(adapter: adapter)
    /// ```
    func unload(adapter: ModelAdapter) {
        adapter.unload(from: self)
    }
    
    /// Fuses an adapter permanently into the model.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try model.fuse(with: adapter)
    /// ```
    func fuse(with adapter: ModelAdapter) throws {
        try adapter.fuse(with: self)
    }
    
    /// Temporarily loads an adapter, performs a synchronous action, then unloads the adapter.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try model.perform(with: adapter) {
    ///     generate(inputs: ...)
    /// }
    /// // Adapter is automatically unloaded after execution
    /// ```
    func perform<R>(with adapter: ModelAdapter, perform: () throws -> R) throws -> R {
        defer {
            adapter.unload(from: self)
        }
        try adapter.load(into: self)
        let result = try perform()
        return result
    }
    
    /// Temporarily loads an adapter, performs an asynchronous action, then unloads the adapter.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try await model.perform(with: adapter) {
    ///     await generate(inputs: ...)
    /// }
    /// // Adapter is automatically unloaded after execution
    /// ```
    func perform<R>(with adapter: ModelAdapter, perform: () async throws -> R) async throws -> R {
        defer {
            adapter.unload(from: self)
        }
        try adapter.load(into: self)
        let result = try await perform()
        return result
    }
}
