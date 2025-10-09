// Copyright © 2025 Apple Inc.

import Accelerate
import ArgumentParser
import Foundation
import MLX
import MLXEmbedders
import Tokenizers

struct SearchCommand: AsyncParsableCommand {
    enum SearchError: LocalizedError {
        case indexNotFound(String)
        case invalidIndex

        var errorDescription: String? {
            switch self {
            case .indexNotFound(let path):
                return "Index file not found at \(path)"
            case .invalidIndex:
                return "Index file is empty or malformed"
            }
        }
    }
    static let configuration = CommandConfiguration(
        commandName: "search",
        abstract: "Search an embedding index for the closest matches"
    )

    @OptionGroup var model: ModelArguments
    @OptionGroup var pooling: PoolingArguments

    @Option(name: .shortAndLong, help: "Path to the embedding index JSON file")
    var index: String

    @Option(name: .shortAndLong, help: "Query text to embed and search with")
    var query: String

    @Option(name: .shortAndLong, help: "Number of results to display")
    var top: Int = 5

    func run() async throws {
        let runtime = try await EmbedderTool.loadRuntime(model: model, pooling: pooling)
        let entries = try loadIndex()
        guard !entries.isEmpty else {
            print("Index at \(index) is empty")
            return
        }

        var queryVector = await embedQuery(runtime: runtime)
        guard !queryVector.isEmpty else {
            print("Query produced no tokens")
            return
        }
        
        queryVector = VectorOperations.normalize(queryVector)

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
        let url = URL(fileURLWithPath: index)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SearchError.indexNotFound(url.path)
        }
        let data = try Data(contentsOf: url)
        let entries = try JSONDecoder().decode([IndexEntry].self, from: data)
        guard !entries.isEmpty else { throw SearchError.invalidIndex }

        if let dimension = entries.first?.embedding.count {
            let mismatched = entries.first { $0.embedding.count != dimension }
            if let mismatch = mismatched {
                reportError("Warning: index entry \(mismatch.path) has dimension \(mismatch.embedding.count) vs expected \(dimension)")
            }
        }

        return entries
    }

    private func embedQuery(runtime: EmbedderRuntime) async -> [Float] {
        do {
            let result = try await runtime.embed(texts: [query])

            if let fallbackMessage = result.fallbackDescription {
                reportError(fallbackMessage)
            }

            guard let embedding = result.embeddings.first(where: { $0.index == 0 }) else {
                return []
            }

            return embedding.vector
        } catch {
            reportError("Pooling error: \(error.localizedDescription)")
            return []
        }
    }

    private func rank(entries: [IndexEntry], query: [Float]) -> [(IndexEntry, Float)] {
        var mismatched: [(path: String, dimension: Int)] = []

        if VectorOperations.hasNonFiniteValues(query) {
            reportError("Query vector contains non-finite values; search aborted")
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
            reportError(dimensionMismatchMessage(for: mismatched, expected: query.count))
        }

        return scored
    }

    private func dimensionMismatchMessage(for mismatched: [(path: String, dimension: Int)], expected: Int) -> String {
        var message = "Skipped \(mismatched.count) index entry(s) with dimension mismatch (expected \(expected))"
        if !mismatched.isEmpty {
            let preview = mismatched.prefix(5).map { "\($0.path) (\($0.dimension))" }.joined(separator: ", ")
            message += ": \(preview)"
            if mismatched.count > 5 {
                message += ", ..."
            }
        }
        return message
    }
}
