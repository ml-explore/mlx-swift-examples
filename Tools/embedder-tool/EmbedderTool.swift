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
        subcommands: [IndexCommand.self, SearchCommand.self, ListCommand.self]
    )

    private static let defaultModelConfiguration = ModelConfiguration.gte_tiny

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments

    mutating func run() async throws {
        let runtime = try await Self.loadRuntime(model: model, pooling: pooling)
        print("Loaded \(runtime.configuration.name) using \(runtime.poolingDescription) pooling")
    }

    static func loadRuntime(model: ModelArguments, pooling: PoolingArguments) async throws -> EmbedderRuntime {
        let loadedModel = try await model.load(default: defaultModelConfiguration)
        return EmbedderRuntime(
            configuration: loadedModel.configuration,
            container: loadedModel.container,
            strategyOverride: pooling.strategy,
            normalize: pooling.normalize,
            applyLayerNorm: pooling.layerNorm
        )
    }
}

struct EmbedderRuntime {
    let configuration: ModelConfiguration
    let container: ModelContainer
    let strategyOverride: Pooling.Strategy?
    let normalize: Bool
    let applyLayerNorm: Bool
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
        let result = try CorpusLoader(
            root: corpus.directoryURL,
            extensions: corpus.normalizedExtensions,
            recursive: corpus.recursive,
            limit: corpus.limit
        ).load()

        if !result.failures.isEmpty {
            reportCorpusFailures(result.failures)
        }

        return result.documents
    }

    private func embed(documents: [Document], runtime: EmbedderRuntime) async throws -> [IndexEntry] {
        guard !documents.isEmpty else { return [] }

        let batchSize = 32
        var accumulatedEntries: [IndexEntry] = []
        var skippedDocuments: [String] = []
        var fallbackMessages: Set<String> = []

        var index = 0
        while index < documents.count {
            let upperBound = min(index + batchSize, documents.count)
            let batch = Array(documents[index..<upperBound])
            let result = try await embedBatch(documents: batch, runtime: runtime)
            accumulatedEntries.append(contentsOf: result.entries)
            skippedDocuments.append(contentsOf: result.skipped)
            if let message = result.fallbackMessage {
                fallbackMessages.insert(message)
            }
            index = upperBound
        }

        if !skippedDocuments.isEmpty {
            reportSkippedDocuments(skippedDocuments)
        }

        for message in fallbackMessages {
            reportPoolingFallback(message)
        }

        return accumulatedEntries
    }

    private func embedBatch(
        documents: [Document],
        runtime: EmbedderRuntime
    ) async throws -> (entries: [IndexEntry], skipped: [String], fallbackMessage: String?) {
        guard !documents.isEmpty else { return ([], [], nil) }

        return try await runtime.container.perform { model, tokenizer, pooler in
            var skippedDocuments: [String] = []

            let encoded = documents.compactMap { document -> (Document, [Int])? in
                let tokens = tokenizer.encode(text: document.contents, addSpecialTokens: true)
                guard !tokens.isEmpty else {
                    skippedDocuments.append(document.path)
                    return nil
                }
                return (document, tokens)
            }

            guard !encoded.isEmpty else { return ([IndexEntry](), skippedDocuments, nil) }

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

            let poolingModule = PoolingSupport.resolvedPooler(base: pooler, runtime: runtime)
            let pooled = poolingModule(
                outputs,
                mask: mask,
                normalize: runtime.normalize,
                applyLayerNorm: runtime.applyLayerNorm
            )
            pooled.eval()
            let extraction = try PoolingSupport.extractVectors(
                from: pooled,
                expectedCount: encoded.count,
                runtime: runtime
            )

            let entries: [IndexEntry] = zip(encoded.map { $0.0 }, extraction.vectors).map { document, vector in
                IndexEntry(path: document.path, embedding: vector)
            }
            return (entries, skippedDocuments, extraction.fallbackDescription)
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

    private func reportSkippedDocuments(_ paths: [String]) {
        var message = "Skipped \(paths.count) document(s) that produced no tokens"
        if !paths.isEmpty {
            let preview = paths.prefix(5).joined(separator: ", ")
            message += ": \(preview)"
            if paths.count > 5 {
                message += ", ..."
            }
        }
        if let data = (message + "\n").data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    private func reportPoolingFallback(_ message: String) {
        if let data = (message + "\n").data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    private func reportCorpusFailures(_ failures: [(url: URL, error: CorpusLoader.ReadError)]) {
        var message = "Skipped \(failures.count) file(s) while reading corpus"
        if !failures.isEmpty {
            let preview = failures.prefix(5).map { failure in
                let description = failure.error.errorDescription ?? "unreadable"
                return "\(failure.url.lastPathComponent) (\(description))"
            }.joined(separator: ", ")
            message += ": \(preview)"
            if failures.count > 5 {
                message += ", ..."
            }
        }
        if let data = (message + "\n").data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

