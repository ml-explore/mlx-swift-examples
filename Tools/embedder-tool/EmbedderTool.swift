// Copyright © 2025 Apple Inc.

import ArgumentParser
import MLXEmbedders

@main
struct EmbedderTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for working with MLX embedders"
    )

    private static let defaultModelConfiguration = ModelConfiguration.gte_tiny

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments

    mutating func run() async throws {
        let runtime = try await loadRuntime()
        print("Loaded \(runtime.configuration.name) using \(runtime.poolingStrategy) pooling")
    }

    private func loadRuntime() async throws -> EmbedderRuntime {
        let loadedModel = try await model.load(default: Self.defaultModelConfiguration)
        return EmbedderRuntime(
            configuration: loadedModel.configuration,
            container: loadedModel.container,
            poolingStrategy: pooling.strategy,
            normalize: pooling.normalize
        )
    }
}

struct EmbedderRuntime {
    let configuration: ModelConfiguration
    let container: ModelContainer
    let poolingStrategy: Pooling.Strategy
    let normalize: Bool
}

