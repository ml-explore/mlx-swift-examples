// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXEmbedders
import Tokenizers

struct SearchCommand: AsyncParsableCommand {
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

        let queryVector = await embedQuery(runtime: runtime)
        guard !queryVector.isEmpty else {
            print("Query produced no tokens")
            return
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
        let url = URL(fileURLWithPath: index)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([IndexEntry].self, from: data)
    }

    private func embedQuery(runtime: EmbedderRuntime) async -> [Float] {
        await runtime.container.perform { model, tokenizer, pooler in
            let tokens = tokenizer.encode(text: query, addSpecialTokens: true)
            guard !tokens.isEmpty else { return [] }

            let padToken = tokenizer.eosTokenId ?? 0
            let maxLength = max(tokens.count, 1)

            let padded = stacked([
                MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
            ])
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
            let vectors: [[Float]] = pooled.map { $0.asArray(Float.self) }
            return vectors.first ?? []
        }
    }

    private func rank(entries: [IndexEntry], query: [Float]) -> [(IndexEntry, Float)] {
        entries.compactMap { entry in
            guard entry.embedding.count == query.count else { return nil }
            let score = cosineSimilarity(entry.embedding, query)
            return (entry, score)
        }
        .sorted { $0.1 > $1.1 }
    }

    private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        var dot: Float = 0
        var lhsNorm: Float = 0
        var rhsNorm: Float = 0

        for (l, r) in zip(lhs, rhs) {
            dot += l * r
            lhsNorm += l * l
            rhsNorm += r * r
        }

        let denominator = sqrt(lhsNorm) * sqrt(rhsNorm)
        guard denominator > 0 else { return 0 }
        return dot / denominator
    }
}
