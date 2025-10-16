// Copyright © 2025 Apple Inc.

import ArgumentParser
import MLXEmbedders

struct PoolingArguments: ParsableArguments {

    @Option(
        name: .long, help: "Pooling strategy used to collapse token embeddings (default: mean).")
    var strategy: Pooling.Strategy?

    @Flag(
        name: .long, inversion: .prefixedNo,
        help:
            "Normalize pooled embeddings to unit length (default: true). Use --no-normalize to disable."
    )
    var normalize = true

    @Flag(name: .long, help: "Apply layer normalization before pooling.")
    var layerNorm = false
}

extension PoolingArguments {
    var strategyOverride: Pooling.Strategy? {
        strategy ?? .mean
    }
}

extension Pooling.Strategy: @retroactive CaseIterable {
    public static var allCases: [Pooling.Strategy] {
        [.mean, .cls, .first, .last, .max, .none]
    }
}

extension Pooling.Strategy: @retroactive ExpressibleByArgument {
    public init?(argument: String) {
        switch argument.lowercased() {
        case "mean": self = .mean
        case "cls": self = .cls
        case "first": self = .first
        case "last": self = .last
        case "max": self = .max
        case "none": self = .none
        default: return nil
        }
    }
}
