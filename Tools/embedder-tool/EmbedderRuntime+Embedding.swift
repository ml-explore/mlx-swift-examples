import Foundation
import MLX
import MLXEmbedders
import Tokenizers

public struct RuntimeEmbeddingResult {
    public let embeddings: [(index: Int, vector: [Float])]
    public let skippedIndices: [Int]
    public let fallbackDescription: String?

    public init(
        embeddings: [(index: Int, vector: [Float])],
        skippedIndices: [Int],
        fallbackDescription: String?
    ) {
        self.embeddings = embeddings
        self.skippedIndices = skippedIndices
        self.fallbackDescription = fallbackDescription
    }
}

extension EmbedderRuntime {
    func embed(texts: [String]) async throws -> RuntimeEmbeddingResult {
        guard !texts.isEmpty else {
            return RuntimeEmbeddingResult(
                embeddings: [], skippedIndices: [], fallbackDescription: nil)
        }

        return try await container.perform { model, tokenizer, pooler in
            var skippedIndices: [Int] = []

            let encoded = texts.enumerated().compactMap { index, text -> (Int, [Int])? in
                let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
                guard !tokens.isEmpty else {
                    skippedIndices.append(index)
                    return nil
                }
                return (index, tokens)
            }

            guard !encoded.isEmpty else {
                return RuntimeEmbeddingResult(
                    embeddings: [],
                    skippedIndices: skippedIndices,
                    fallbackDescription: nil
                )
            }

            guard let padToken = tokenizer.eosTokenId else {
                throw CommandError("Could not determine a padding token from the tokenizer.")
            }
            let maxLength = encoded.map { $0.1.count }.max() ?? 0

            let padded = stacked(
                encoded.map { _, tokens in
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

            let poolingModule = resolvedPooler(for: pooler)
            let pooled = poolingModule(
                outputs,
                mask: mask,
                normalize: self.normalize,
                applyLayerNorm: self.applyLayerNorm
            )
            pooled.eval()

            let extraction = try extractVectors(from: pooled, expectedCount: encoded.count)

            let embeddings = zip(encoded.map { $0.0 }, extraction.vectors).map { index, vector in
                (index: index, vector: vector)
            }

            return RuntimeEmbeddingResult(
                embeddings: embeddings,
                skippedIndices: skippedIndices,
                fallbackDescription: extraction.fallbackDescription
            )
        }
    }
}
