// Copyright © 2025 Apple Inc.

import ArgumentParser
import MLXEmbedders

struct PoolingArguments: ParsableArguments {

    @Option(name: .long, help: "Pooling strategy used to collapse token embeddings into a single vector.")
    var strategy: Pooling.Strategy = .mean

    @Flag(name: .long, help: "Normalize pooled embeddings to unit length.")
    var normalize = false
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
