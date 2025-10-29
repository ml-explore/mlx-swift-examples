// Copyright © 2025 Apple Inc.

import Accelerate
import ArgumentParser
import Foundation
import MLX
import MLXEmbedders
import Tokenizers

struct SearchCommand: EmbedderCommand {
    static let configuration = CommandConfiguration(
        commandName: "search",
        abstract: "Search an embedding index for the closest matches"
    )

    @OptionGroup var model: ModelArguments
    @OptionGroup var pooling: PoolingArguments
    @OptionGroup var memory: MemoryArguments

    @Option(name: .shortAndLong, help: "Path to the embedding index JSON file")
    var index: URL

    @Option(name: .shortAndLong, help: "Query text to embed and search with")
    var query: String

    @Option(name: .shortAndLong, help: "Number of results to display")
    var top: Int = 5

    mutating func run(runtime: EmbedderRuntime) async throws {
        let entries = try loadIndex()
        guard !entries.isEmpty else {
            print("Index at \(index.path) is empty")
            return
        }

        var queryVector = await embedQuery(runtime: runtime)
        guard !queryVector.isEmpty else {
            print("Query produced no tokens")
            return
        }

        queryVector = VectorOperations.sanitize(queryVector)

        if runtime.normalize {
            queryVector = VectorOperations.normalize(queryVector)
        }

        let results = rank(entries: entries, query: queryVector)
        if results.isEmpty {
            print("No comparable embeddings in index")
            return
        }

        let limit = max(0, min(top, results.count))
        for (entry, score) in results.prefix(limit) {
            print(String(format: "%@\t%.4f", entry.path, score))
        }
    }

    private func loadIndex() throws -> [IndexEntry] {
        guard FileManager.default.fileExists(atPath: index.path) else {
            throw CommandError("Index file not found at \(index.path)")
        }
        let data = try Data(contentsOf: index)
        let entries = try JSONDecoder().decode([IndexEntry].self, from: data)
        guard !entries.isEmpty else { throw CommandError("Index file is empty or malformed") }

        if let dimension = entries.first?.embedding.count {
            let mismatched = entries.first { $0.embedding.count != dimension }
            if let mismatch = mismatched {
                writeDiagnostic(
                    "Index entry \(mismatch.path) has dimension \(mismatch.embedding.count) vs expected \(dimension)",
                    kind: .warning)
            }
        }

        return entries
    }

    private func embedQuery(runtime: EmbedderRuntime) async -> [Float] {
        do {
            let result = try await runtime.embed(texts: [query])

            if let fallbackMessage = result.fallbackDescription {
                writeDiagnostic(fallbackMessage, kind: .warning)
            }

            guard let embedding = result.embeddings.first(where: { $0.index == 0 }) else {
                return []
            }

            return embedding.vector
        } catch {
            writeDiagnostic("Pooling error: \(error.localizedDescription)", kind: .error)
            return []
        }
    }

    private func rank(entries: [IndexEntry], query: [Float]) -> [(IndexEntry, Float)] {
        var mismatched: [(path: String, dimension: Int)] = []

        if VectorOperations.hasNonFiniteValues(query) {
            writeDiagnostic("Query vector contains non-finite values; search aborted", kind: .error)
            return []
        }

        let scored = entries.compactMap { entry -> (IndexEntry, Float)? in
            let dimension = entry.embedding.count
            guard dimension == query.count else {
                mismatched.append((entry.path, dimension))
                return nil
            }
            guard !VectorOperations.hasNonFiniteValues(entry.embedding) else {
                return (entry, 0)
            }
            let score = VectorOperations.dotProduct(entry.embedding, query)
            return (entry, score)
        }
        .sorted { $0.1 > $1.1 }

        if !mismatched.isEmpty {
            writeDiagnostic(
                dimensionMismatchMessage(for: mismatched, expected: query.count), kind: .warning)
        }

        return scored
    }

    private func dimensionMismatchMessage(
        for mismatched: [(path: String, dimension: Int)], expected: Int
    ) -> String {
        var message =
            "Skipped \(mismatched.count) index entry(s) with dimension mismatch (expected \(expected))"
        if !mismatched.isEmpty {
            let preview = mismatched.prefix(5).map { "\($0.path) (\($0.dimension))" }.joined(
                separator: ", ")
            message += ": \(preview)"
            if mismatched.count > 5 {
                message += ", ..."
            }
        }
        return message
    }
}
