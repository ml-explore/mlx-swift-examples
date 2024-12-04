// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXLinalg
import MLXNN

public struct PoolingConfiguration: Codable {
    public let dimension: Int
    public let poolingModeClsToken: Bool
    public let poolingModeMeanTokens: Bool
    public let poolingModeMaxTokens: Bool
    public let poolingModeLastToken: Bool

    enum CodingKeys: String, CodingKey {
        case dimension = "word_embedding_dimension"
        case poolingModeClsToken = "pooling_mode_cls_token"
        case poolingModeMeanTokens = "pooling_mode_mean_tokens"
        case poolingModeMaxTokens = "pooling_mode_max_tokens"
        case poolingModeLastToken = "pooling_mode_lasttoken"
    }
}

func loadPooling(modelDirectory: URL) -> Pooling {
    let configurationURL = modelDirectory.appending(components: "1_Pooling", "config.json")
    guard
        let poolingConfig = try? JSONDecoder().decode(
            PoolingConfiguration.self, from: Data(contentsOf: configurationURL))
    else {
        return Pooling(strategy: .none)
    }

    return Pooling(config: poolingConfig)
}

public class Pooling: Module {
    public enum Strategy {
        case mean
        case cls
        case first
        case last
        case max
        case none
    }
    let strategy: Strategy
    let dimension: Int?

    public init(
        strategy: Strategy, dimension: Int? = nil
    ) {
        self.strategy = strategy
        self.dimension = dimension
    }

    public init(
        config: PoolingConfiguration
    ) {
        dimension = config.dimension
        if config.poolingModeClsToken {
            strategy = .cls
        } else if config.poolingModeMeanTokens {
            strategy = .mean
        } else if config.poolingModeMaxTokens {
            strategy = .max
        } else if config.poolingModeLastToken {
            strategy = .last
        } else {
            strategy = .first
        }
    }

    public func callAsFunction(
        _ inputs: EmbeddingModelOutput, mask: MLXArray? = nil, normalize: Bool = false,
        applyLayerNorm: Bool = false
    ) -> MLXArray {
        let _mask = mask ?? MLXArray.ones(Array(inputs.hiddenStates?.shape[0 ..< 2] ?? [0]))

        var pooled: MLXArray
        switch self.strategy {
        case .mean:
            pooled =
                sum(
                    inputs.hiddenStates! * _mask.expandedDimensions(axes: [-1]),
                    axis: 1)
                / sum(_mask, axis: -1, keepDims: true)
        case .max:
            pooled = MLX.max(
                inputs.hiddenStates! * _mask.expandedDimensions(axes: [-1]), axis: 1)
        case .first:
            pooled = inputs.hiddenStates![0..., 0, 0...]
        case .last:
            pooled = inputs.hiddenStates![0..., -1, 0...]
        case .cls:
            pooled =
                inputs.pooledOutput
                ?? inputs.hiddenStates![0..., 0, 0...]
        case .none:
            pooled = inputs.pooledOutput ?? inputs.hiddenStates!
        }
        if applyLayerNorm {
            pooled = MLXFast.layerNorm(pooled, eps: 1e-5)
        }
        if let dimension {
            pooled = pooled[0..., 0 ..< dimension]
        }
        if normalize {
            pooled = pooled / norm(pooled, axis: -1, keepDims: true)
        }
        return pooled
    }
}
