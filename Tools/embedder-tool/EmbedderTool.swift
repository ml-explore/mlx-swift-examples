// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXEmbedders
import Tokenizers

@main
struct EmbedderTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for working with MLX embedders",
        subcommands: [IndexCommand.self]
    )

    private static let defaultModelConfiguration = ModelConfiguration.gte_tiny

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments

    mutating func run() async throws {
        let runtime = try await Self.loadRuntime(model: model, pooling: pooling)
        print("Loaded \(runtime.configuration.name) using \(runtime.poolingStrategy) pooling")
    }

    static func loadRuntime(model: ModelArguments, pooling: PoolingArguments) async throws -> EmbedderRuntime {
        let loadedModel = try await model.load(default: defaultModelConfiguration)
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

struct IndexCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "index",
        abstract: "Create an embedding index for a corpus"
    )

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments

    @Option(name: .shortAndLong, help: "Destination file for the generated index")
    var output: String

    func run() async throws {
        let runtime = try await EmbedderTool.loadRuntime(model: model, pooling: pooling)
        let documents = try loadDocuments()
        let entries = try await embed(documents: documents, runtime: runtime)
        try writeIndex(entries: entries, to: outputURL)
        print("Wrote \(entries.count) embeddings to \(outputURL.path)")
    }

    private func loadDocuments() throws -> [Document] {
        try CorpusLoader(
            root: corpus.directoryURL,
            extensions: corpus.normalizedExtensions,
            recursive: corpus.recursive,
            limit: corpus.limit
        ).load()
    }

    private func embed(documents: [Document], runtime: EmbedderRuntime) async throws -> [IndexEntry] {
        guard !documents.isEmpty else { return [] }

        return try await runtime.container.perform { model, tokenizer, pooler in
            let encoded = documents.compactMap { document -> (Document, [Int])? in
                let tokens = tokenizer.encode(text: document.contents, addSpecialTokens: true)
                guard !tokens.isEmpty else { return nil }
                return (document, tokens)
            }

            guard !encoded.isEmpty else { return [] }

            let padToken = tokenizer.eosTokenId ?? 0
            let maxLength = encoded.map { $0.1.count }.max() ?? 0

            let padded = stacked(encoded.map { _, tokens in
                MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
            })
            let mask = (padded .!= padToken)
            let tokenTypes = MLXArray.zeros(like: padded)
            let outputs = model(
                padded,
                positionIds: nil,
                tokenTypeIds: tokenTypes,
                attentionMask: mask
            )

            let poolingModule: Pooling = {
                if runtime.poolingStrategy == .none {
                    return pooler
                } else {
                    return Pooling(strategy: runtime.poolingStrategy)
                }
            }()

            let pooled = poolingModule(outputs, mask: mask, normalize: runtime.normalize)
            pooled.eval()
            let flattened: [Float] = pooled.asArray(Float.self)
            let vectorCount = encoded.count
            let dimension = vectorCount == 0 ? 0 : flattened.count / vectorCount
            guard dimension > 0 else { return [] }

            var vectors: [[Float]] = []
            vectors.reserveCapacity(vectorCount)
            for index in 0..<vectorCount {
                let start = index * dimension
                let end = start + dimension
                vectors.append(Array(flattened[start..<end]))
            }

            return zip(encoded.map { $0.0 }, vectors).map { document, vector in
                IndexEntry(path: document.path, embedding: vector)
            }
        }
    }

    private func writeIndex(entries: [IndexEntry], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(entries)
        try data.write(to: url)
    }

    private var outputURL: URL {
        URL(fileURLWithPath: output)
    }
}

