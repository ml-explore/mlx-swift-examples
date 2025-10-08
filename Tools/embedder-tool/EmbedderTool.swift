// Copyright © 2025 Apple Inc.

import ArgumentParser

@main
struct EmbedderTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for working with MLX embedders"
    )

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments

    mutating func run() async throws {
        // Implementation will be expanded in following steps.
    }
}

