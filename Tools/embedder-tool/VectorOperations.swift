// Copyright © 2025 Apple Inc.

import Accelerate
import Foundation

enum VectorOperations {
    static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return 0 }
        guard !lhs.isEmpty else { return 0 }

        var dot: Float = 0
        var lhsNormSquared: Float = 0
        var rhsNormSquared: Float = 0

        vDSP_dotpr(lhs, 1, rhs, 1, &dot, vDSP_Length(lhs.count))
        vDSP_svesq(lhs, 1, &lhsNormSquared, vDSP_Length(lhs.count))
        vDSP_svesq(rhs, 1, &rhsNormSquared, vDSP_Length(rhs.count))

        let denominator = sqrt(lhsNormSquared * rhsNormSquared)
        guard denominator > 1e-9 else { return 0 }
        return dot / denominator
    }

    static func l2Norm(_ vector: [Float]) -> Float {
        guard !vector.isEmpty else { return 0 }
        var sumSquares: Float = 0
        vDSP_svesq(vector, 1, &sumSquares, vDSP_Length(vector.count))
        return sqrt(sumSquares)
    }

    static func normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }

        let sanitized = sanitize(vector)

        var sumSquares: Float = 0
        vDSP_svesq(sanitized, 1, &sumSquares, vDSP_Length(sanitized.count))

        guard sumSquares.isFinite else { return [] }
        guard sumSquares > 1e-9 else { return sanitized }

        var divisor = sqrt(sumSquares)
        var normalized = [Float](repeating: 0, count: sanitized.count)
        vDSP_vsdiv(sanitized, 1, &divisor, &normalized, 1, vDSP_Length(sanitized.count))

        return normalized
    }

    static func sanitize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }

        var sanitized = vector
        for index in sanitized.indices where !sanitized[index].isFinite {
            sanitized[index] = 0
        }
        return sanitized
    }

    static func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return 0 }
        guard !lhs.isEmpty else { return 0 }

        var result: Float = 0
        vDSP_dotpr(lhs, 1, rhs, 1, &result, vDSP_Length(lhs.count))
        return result
    }

    static func normalizeBatch(_ vectors: [[Float]]) -> [[Float]] {
        vectors.map { normalize($0) }
    }

    static func batchCosineSimilarity(query: [Float], documents: [[Float]]) -> [Float] {
        documents.map { cosineSimilarity(query, $0) }
    }

    static func batchDotProduct(query: [Float], documents: [[Float]]) -> [Float] {
        documents.map { dotProduct(query, $0) }
    }
}

extension VectorOperations {
    static func hasNonFiniteValues(_ vector: [Float]) -> Bool {
        vector.contains(where: { !$0.isFinite })
    }

    static func statistics(_ vector: [Float]) -> (mean: Float, min: Float, max: Float, norm: Float)
    {
        guard !vector.isEmpty else { return (0, 0, 0, 0) }

        var mean: Float = 0
        var min: Float = 0
        var max: Float = 0

        vDSP_meanv(vector, 1, &mean, vDSP_Length(vector.count))
        vDSP_minv(vector, 1, &min, vDSP_Length(vector.count))
        vDSP_maxv(vector, 1, &max, vDSP_Length(vector.count))

        return (mean, min, max, l2Norm(vector))
    }
}
