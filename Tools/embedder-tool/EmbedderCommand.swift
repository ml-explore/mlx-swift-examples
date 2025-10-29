// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation

/// A protocol for commands that need to load and use an embedder model.
///
/// This protocol centralizes the logic for loading an `EmbedderRuntime`
/// and managing memory statistics, reducing boilerplate in individual commands.
protocol EmbedderCommand: AsyncParsableCommand {
    /// The model arguments, captured via `@OptionGroup`.
    var model: ModelArguments { get }

    /// The pooling arguments, captured via `@OptionGroup`.
    var pooling: PoolingArguments { get }

    /// The memory management arguments, captured via `@OptionGroup`.
    var memory: MemoryArguments { get set }

    /// The core logic of the command, which receives a fully initialized `EmbedderRuntime`.
    /// - Parameter runtime: The loaded and configured embedder runtime.
    mutating func run(runtime: EmbedderRuntime) async throws
}

extension EmbedderCommand {
    /// The main entry point for the command.
    ///
    /// This default implementation handles the loading of the embedder runtime
    /// and memory reporting, then calls the command's specific `run(runtime:)` method.
    mutating func run() async throws {
        var memory = self.memory
        let capturedModel = model
        let capturedPooling = pooling

        let runtime = try await memory.start {
            try await EmbedderTool.loadRuntime(model: capturedModel, pooling: capturedPooling)
        }

        defer {
            memory.reportMemoryStatistics()
            self.memory = memory
        }

        try await run(runtime: runtime)
    }
}
