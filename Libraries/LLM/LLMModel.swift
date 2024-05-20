// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// Interface for all LLM Models
public protocol LLMModel: Module {

    var vocabularySize: Int { get }

    func callAsFunction(_ inputs: MLXArray, cache: [(MLXArray, MLXArray)]?) -> (
        MLXArray, [(MLXArray, MLXArray)]
    )

    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

extension LLMModel {

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

}
