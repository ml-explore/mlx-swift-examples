// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/clip.py

struct CLIPOutput {
    /// The lastHiddenState indexed at the EOS token and possibly projected if
    /// the model has a projection layer
    public var pooledOutput: MLXArray

    /// The full sequence output of the transformer after the final layernorm
    public var lastHiddenState: MLXArray

    /// A list of hidden states corresponding to the outputs of the transformer layers
    public var hiddenStates: [MLXArray]
}

/// The transformer encoder layer from CLIP
class CLIPEncoderLayer: Module {

    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    let attention: MultiHeadAttention

    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    let activation: (MLXArray) -> MLXArray

    init(modelDimensions: Int, numHeads: Int, activation: @escaping (MLXArray) -> MLXArray) {
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: modelDimensions)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: modelDimensions)

        self.attention = MultiHeadAttention(
            dimensions: modelDimensions, numHeads: numHeads, bias: true)

        self.linear1 = Linear(modelDimensions, 4 * modelDimensions)
        self.linear2 = Linear(4 * modelDimensions, modelDimensions)

        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var y = layerNorm1(x)
        y = attention(y, keys: y, values: y, mask: attentionMask)
        var x = y + x

        y = layerNorm2(x)
        y = linear1(y)
        y = activation(y)
        y = linear2(y)
        x = y + x

        return x
    }
}

/// Implements the text encoder transformer from CLIP
class CLIPTextModel: Module {

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let layers: [CLIPEncoderLayer]

    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    @ModuleInfo(key: "text_projection") var textProjection: Linear?

    init(configuration: CLIPTextModelConfiguration) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.modelDimensions)
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.maxLength, dimensions: configuration.modelDimensions)

        self.layers = (0 ..< configuration.numLayers)
            .map { _ in
                CLIPEncoderLayer(
                    modelDimensions: configuration.modelDimensions,
                    numHeads: configuration.numHeads,
                    activation: configuration.hiddenActivation.activation)
            }

        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: configuration.modelDimensions)

        if let projectionDimensions = configuration.projectionDimensions {
            self._textProjection.wrappedValue = Linear(
                configuration.modelDimensions, projectionDimensions, bias: false)
        } else {
            self._textProjection.wrappedValue = nil
        }
    }

    func mask(_ N: Int, _ dType: DType) -> MLXArray {
        let indices = MLXArray(0 ..< Int32(N))
        var mask = indices[0..., .newAxis] .< indices[.newAxis]
        mask = mask.asType(dType) * (dType == .float16 ? -6e4 : -1e9)
        return mask
    }

    func callAsFunction(_ x: MLXArray) -> CLIPOutput {
        var x = x
        let (_, N) = x.shape2
        let eosTokens = x.argMax(axis: -1)

        // compute the embeddings
        x = tokenEmbedding(x)
        x = x + positionEmbedding.weight[..<N]

        // compute the features from the transformer
        let mask = mask(N, x.dtype)
        var hiddenStates = [MLXArray]()
        for l in layers {
            x = l(x, attentionMask: mask)
            hiddenStates.append(x)
        }

        // apply the final layernorm
        x = finalLayerNorm(x)
        let lastHiddenState = x

        // select the EOS token
        var pooledOutput = x[MLXArray(0 ..< x.count), eosTokens]
        if let textProjection {
            pooledOutput = textProjection(pooledOutput)
        }

        return CLIPOutput(
            pooledOutput: pooledOutput, lastHiddenState: lastHiddenState, hiddenStates: hiddenStates
        )
    }
}
