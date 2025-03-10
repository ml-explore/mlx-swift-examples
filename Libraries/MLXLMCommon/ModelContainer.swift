// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``ModelContext``):
///
/// ```swift
/// let messages = [["role": "user", "content": prompt]]
/// let promptTokens = try await modelContainer.perform { context in
///     try context.tokenizer.applyChatTemplate(messages: messages)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let userInput: UserInput
/// let result = await modelContainer.perform { context in
///     let input = try await context.processor.prepare(input: userInput)
///     return generate(
///         input: input, parameters: generateParameters, context: context
///     ) { tokens in
///     ...
///     }
/// }
/// ```
public actor ModelContainer {
    var context: ModelContext
    public var configuration: ModelConfiguration { context.configuration }

    public init(context: ModelContext) {
        self.context = context
    }

    /// Perform an action on the model and/or tokenizer.  Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R>(_ action: @Sendable (any LanguageModel, Tokenizer) throws -> R) rethrows
        -> R
    {
        try action(context.model, context.tokenizer)
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V, R>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> R
    ) rethrows -> R {
        try action(context.model, context.tokenizer, values)
    }

    /// Perform an action on the ``ModelContext``.  Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<R>(_ action: @Sendable (ModelContext) async throws -> R) async rethrows -> R
    {
        try await action(context)
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> R {
        try await action(context, values)
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout ModelContext) -> Void) {
        action(&context)
    }
}
