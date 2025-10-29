import Foundation
import MLX
import MLXEmbedders

enum PoolingError: LocalizedError {
    case unsupportedShape([Int])
    case vectorCountMismatch(expected: Int, received: Int)

    var errorDescription: String? {
        switch self {
        case .unsupportedShape(let shape):
            return "Pooling produced unsupported shape: \(shape)"
        case .vectorCountMismatch(let expected, let received):
            return "Pooling produced \(received) vectors but expected \(expected)"
        }
    }
}

struct PoolingExtraction {
    let vectors: [[Float]]
    let fallbackDescription: String?
}

extension EmbedderRuntime {
    func resolvedPooler(for pooler: Pooling) -> Pooling {
        guard let override = strategyOverride else {
            return pooler
        }

        if pooler.strategy == override {
            return pooler
        }

        if let dimension = pooler.dimension {
            return Pooling(strategy: override, dimension: dimension)
        } else {
            return Pooling(strategy: override)
        }
    }

    func extractVectors(from array: MLXArray, expectedCount: Int) throws -> PoolingExtraction {
        let shape = array.shape

        switch shape.count {
        case 2:
            let vectors = array.map { $0.asArray(Float.self) }
            guard vectors.count == expectedCount else {
                throw PoolingError.vectorCountMismatch(
                    expected: expectedCount, received: vectors.count)
            }
            return PoolingExtraction(vectors: vectors, fallbackDescription: nil)

        case 3:
            let reduced = mean(array, axis: 1)
            reduced.eval()
            let vectors = reduced.map { $0.asArray(Float.self) }
            guard vectors.count == expectedCount else {
                throw PoolingError.vectorCountMismatch(
                    expected: expectedCount, received: vectors.count)
            }

            let effectiveStrategy = strategyOverride ?? baseStrategy
            let description: String
            if effectiveStrategy == .none {
                description =
                    "Pooling strategy 'none' returned sequence embeddings; falling back to mean over tokens."
            } else {
                description =
                    "Pooling returned sequence embeddings; falling back to mean over tokens."
            }
            return PoolingExtraction(vectors: vectors, fallbackDescription: description)

        default:
            throw PoolingError.unsupportedShape(shape)
        }
    }
}

extension Pooling.Strategy {
    var cliDescription: String {
        switch self {
        case .mean: return "mean"
        case .cls: return "cls"
        case .first: return "first"
        case .last: return "last"
        case .max: return "max"
        case .none: return "none"
        }
    }
}

extension EmbedderRuntime {
    var poolingDescription: String {
        if let override = strategyOverride {
            return "override (\(override.cliDescription))"
        } else {
            return "model default (\(baseStrategy.cliDescription))"
        }
    }
}
