// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// Interface for all LLM Models
public protocol LLMModel: Module {
    func callAsFunction(_ inputs: MLXArray, cache: [(MLXArray, MLXArray)]?) -> (
        MLXArray, [(MLXArray, MLXArray)]
    )
}
