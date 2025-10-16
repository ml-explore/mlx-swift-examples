// Copyright © 2025 Apple Inc.

import Accelerate
import ArgumentParser
import Foundation
import MLX
import MLXEmbedders
import Tokenizers

@main
struct EmbedderTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for working with MLX embedders",
        subcommands: [
            IndexCommand.self, SearchCommand.self, ReplCommand.self, ListCommand.self,
            DemoCommand.self,
        ]
    )

    private static let defaultModelConfiguration = ModelConfiguration.nomic_text_v1_5

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments
    @OptionGroup var memory: MemoryArguments

    mutating func run() async throws {
        var memory = self.memory
        let capturedModel = model
        let capturedPooling = pooling
        let runtime = try await memory.start {
            try await Self.loadRuntime(model: capturedModel, pooling: capturedPooling)
        }
        defer {
            memory.reportMemoryStatistics()
            self.memory = memory
        }
        print("Loaded \(runtime.configuration.name) using \(runtime.poolingDescription) pooling")
    }

    static func loadRuntime(model: ModelArguments, pooling: PoolingArguments) async throws
        -> EmbedderRuntime
    {
        let loadedModel = try await model.load(default: defaultModelConfiguration)
        let baseStrategy = await loadedModel.container.perform { _, _, pooler in
            pooler.strategy
        }
        return EmbedderRuntime(
            configuration: loadedModel.configuration,
            container: loadedModel.container,
            baseStrategy: baseStrategy,
            strategyOverride: pooling.strategyOverride,
            normalize: pooling.normalize,
            applyLayerNorm: pooling.layerNorm
        )
    }
}

struct EmbedderRuntime {
    let configuration: ModelConfiguration
    let container: ModelContainer
    let baseStrategy: Pooling.Strategy
    let strategyOverride: Pooling.Strategy?
    let normalize: Bool
    let applyLayerNorm: Bool
}

struct IndexCommand: EmbedderCommand {
    static let configuration = CommandConfiguration(
        commandName: "index",
        abstract: "Create an embedding index for a corpus"
    )

    @OptionGroup var model: ModelArguments
    @OptionGroup var pooling: PoolingArguments
    @OptionGroup var memory: MemoryArguments

    @OptionGroup var corpus: CorpusArguments

    @Option(name: .shortAndLong, help: "Destination file for the generated index")
    var output: URL

    @Option(name: .long, help: "Number of documents to embed per batch (default: 8)")
    var batchSize: Int = 8

    mutating func run(runtime: EmbedderRuntime) async throws {
        let documents = try loadDocuments()
        let entries = try await embed(documents: documents, runtime: runtime, batchSize: batchSize)
        try writeIndex(entries: entries, to: output)
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

    private func embed(documents: [Document], runtime: EmbedderRuntime, batchSize: Int) async throws
        -> [IndexEntry]
    {
        guard !documents.isEmpty else { return [] }

        let batchSize = max(1, min(batchSize, documents.count))
        var accumulatedEntries: [IndexEntry] = []
        var skippedDocuments: [String] = []
        var fallbackMessages: Set<String> = []
        var lastReportedMilestone: Int = -1

        var index = 0
        while index < documents.count {
            let upperBound = min(index + batchSize, documents.count)
            let batch = Array(documents[index ..< upperBound])
            let result = try await runtime.embed(texts: batch.map { $0.contents })

            let entries: [IndexEntry] = result.embeddings.compactMap { embedding in
                guard batch.indices.contains(embedding.index) else { return nil }
                let document = batch[embedding.index]
                let vector =
                    runtime.normalize
                    ? VectorOperations.normalize(embedding.vector)
                    : VectorOperations.sanitize(embedding.vector)
                return IndexEntry(path: document.path, embedding: vector)
            }
            accumulatedEntries.append(contentsOf: entries)

            skippedDocuments.append(
                contentsOf: result.skippedIndices.compactMap { index -> String? in
                    guard batch.indices.contains(index) else { return nil }
                    return batch[index].path
                })

            if let message = result.fallbackDescription {
                fallbackMessages.insert(message)
            }
            if documents.count > 0 {
                let milestone = Int((Double(upperBound) / Double(documents.count)) * 10.0)
                if milestone > lastReportedMilestone || upperBound == documents.count {
                    reportProgress(processed: upperBound, total: documents.count)
                    lastReportedMilestone = milestone
                }
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

    private func writeIndex(entries: [IndexEntry], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(entries)
        try data.write(to: url)
    }

    private func reportProgress(processed: Int, total: Int) {
        guard total > 0 else { return }
        let message = "Processed \(processed)/\(total) documents"
        writeDiagnostic(message, kind: .info)
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
        writeDiagnostic(message, kind: .warning)
    }

    private func reportPoolingFallback(_ message: String) {
        writeDiagnostic(message, kind: .warning)
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
        writeDiagnostic(message, kind: .warning)
    }
}
