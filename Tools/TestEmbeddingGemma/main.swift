import MLX
import MLXEmbedders
import MLXLMCommon
import MLXLLM
import Foundation
import Hub
import Hub
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Cmlx
import ArgumentParser
@preconcurrency import Tokenizers

import Foundation

extension Tokenizer {
    func tokenize(_ strings: [String]) -> (MLXArray, MLXArray) {
        let tokenized = strings.map {self($0)}
        let maxCount = tokenized.map(\.count).max()!
        let padded = stacked(tokenized.map {
            MLXArray($0 + Array(repeating: 0, count: maxCount - $0.count))
        })
        let mask = stacked(tokenized.map {
            let basicMask = MLXArray.zeros([maxCount, maxCount], dtype: .bool)
            basicMask[0..., ..<$0.count] = MLXArray(true)
            return basicMask
        }).reshaped(padded.shape[0], 1, padded.shape[1], padded.shape[1])
        return (padded, mask)
    }
}

@main
struct Run: AsyncParsableCommand {
    mutating func run() async throws {
        let configurations = [
            ModelConfiguration.embeddinggemma_300m,
            ModelConfiguration.embeddinggemma_300m_8bit,
            ModelConfiguration.embeddinggemma_300m_6bit,
            ModelConfiguration.embeddinggemma_300m_4bit
        ]

        for config in configurations {
            print("Testing \(config.name)...")
            let (model, tokenizer) = try await load(configuration: config)

            let (tokens, mask) = tokenizer.tokenize([
                "the cat smells of farts",
                "the dog smells the cat",
                "the dog smells like the cat",
                "the car is not a train"
            ])

            let out: EmbeddingModelOutput = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: mask)
            let sim = matmul(out.pooledOutput!, out.pooledOutput!.transposed())
            let time = ContinuousClock().measure {
                for _ in 0..<100 {
                    let a = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: mask)
                    eval(a.pooledOutput)
                }
            }
            print(sim)

            if #available(macOS 15, *) {
                print(Double(time.attoseconds)/(100*1e18))
            }
        }
    }
}


