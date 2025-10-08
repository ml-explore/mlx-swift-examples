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

enum PoolingSupport {
    static func resolvedPooler(base pooler: Pooling, runtime: EmbedderRuntime) -> Pooling {
        guard let override = runtime.strategyOverride else {
            return pooler
        }

        if let baseStrategy: Pooling.Strategy = value(for: "strategy", in: pooler), baseStrategy == override {
            return pooler
        }

        let dimension: Int? = value(for: "dimension", in: pooler)
        if let dimension {
            return Pooling(strategy: override, dimension: dimension)
        } else {
            return Pooling(strategy: override)
        }
    }

    static func extractVectors(
        from array: MLXArray,
        expectedCount: Int,
        runtime: EmbedderRuntime
    ) throws -> (vectors: [[Float]], fallbackDescription: String?) {
        let shape = array.shape

        switch shape.count {
        case 2:
            let vectors = array.map { $0.asArray(Float.self) }
            guard vectors.count == expectedCount else {
                throw PoolingError.vectorCountMismatch(expected: expectedCount, received: vectors.count)
            }
            return (vectors, nil)

        case 3:
            let reduced = mean(array, axis: 1)
            reduced.eval()
            let vectors = reduced.map { $0.asArray(Float.self) }
            guard vectors.count == expectedCount else {
                throw PoolingError.vectorCountMismatch(expected: expectedCount, received: vectors.count)
            }

            let description: String
            if runtime.strategyOverride == .none {
                description = "Pooling strategy 'none' returned sequence embeddings; falling back to mean over tokens."
            } else {
                description = "Pooling returned sequence embeddings; falling back to mean over tokens."
            }
            return (vectors, description)

        default:
            throw PoolingError.unsupportedShape(shape)
        }
    }

    private static func value<T>(for key: String, in pooler: Pooling) -> T? {
        guard let child = Mirror(reflecting: pooler).children.first(where: { $0.label == key }) else {
            return nil
        }

        if let value = child.value as? T {
            return value
        }

        let mirror = Mirror(reflecting: child.value)
        if mirror.displayStyle == .optional, let first = mirror.children.first?.value as? T {
            return first
        }

        return nil
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
        strategyOverride?.cliDescription ?? "model default"
    }
}
