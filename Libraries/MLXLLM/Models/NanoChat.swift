//
//  NanoChat.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai 10/15/25.
//
//  Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/nanochat.py
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Helpers

private func functionalRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let meanSquares = mean(x.square(), axis: -1, keepDims: true)
    return x * (meanSquares + eps).rsqrt()
}

private func applySoftcap(_ logits: MLXArray, cap: Float) -> MLXArray {
    guard cap > 0 else { return logits }
    let scale = MLXArray(cap)
    return scale * tanh(logits / scale)
}

// MARK: - Attention

private final class NanoChatAttention: Module {
    let config: NanoChatConfiguration
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "c_q") var wq: Linear
    @ModuleInfo(key: "c_k") var wk: Linear
    @ModuleInfo(key: "c_v") var wv: Linear
    @ModuleInfo(key: "c_proj") var wo: Linear

    private let _ropeFreqs: MLXArray

    init(_ config: NanoChatConfiguration) {
        self.config = config
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.hiddenSize / config.attentionHeads
        precondition(headDim % 2 == 0, "Head dimension must be even for rotary embeddings.")

        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        let halfDim = headDim / 2
        let freqIndices = MLXArray(Array(0 ..< halfDim)).asType(.float32)
        let freqScale = Float(log(Double(config.ropeTheta)) / Double(halfDim))
        self._ropeFreqs = -MLX.exp(freqIndices * freqScale)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(batchSize, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, sequenceLength, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, sequenceLength, numKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        let freqs = _ropeFreqs
        queries = MLXFast.RoPE(
            queries,
            dimensions: headDim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
        keys = MLXFast.RoPE(
            keys,
            dimensions: headDim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )

        queries = functionalRMSNorm(queries, eps: config.rmsNormEps)
        keys = functionalRMSNorm(keys, eps: config.rmsNormEps)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return wo(output)
    }
}

// MARK: - MLP

private final class NanoChatMLP: Module, UnaryLayer {
    let config: NanoChatConfiguration

    @ModuleInfo(key: "c_fc") var fc: Linear
    @ModuleInfo(key: "c_proj") var proj: Linear

    init(_ config: NanoChatConfiguration) {
        self.config = config
        _fc.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _proj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = relu(fc(x))
        return proj(activated * activated)
    }
}

// MARK: - Transformer Block

private final class NanoChatBlock: Module {
    let config: NanoChatConfiguration

    @ModuleInfo(key: "attn") var attention: NanoChatAttention
    @ModuleInfo(key: "mlp") var mlp: NanoChatMLP

    init(_ config: NanoChatConfiguration) {
        self.config = config
        _attention.wrappedValue = NanoChatAttention(config)
        _mlp.wrappedValue = NanoChatMLP(config)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attnOutput = attention(
            functionalRMSNorm(x, eps: config.rmsNormEps), mask: mask, cache: cache)
        let residual = x + attnOutput
        let mlpOutput = mlp(functionalRMSNorm(residual, eps: config.rmsNormEps))
        return residual + mlpOutput
    }
}

// MARK: - Model (inner)

private final class NanoChatModelInner: Module {
    let config: NanoChatConfiguration

    @ModuleInfo(key: "wte") var embedTokens: Embedding
    @ModuleInfo(key: "h") var layers: [NanoChatBlock]

    init(_ config: NanoChatConfiguration) {
        precondition(config.vocabularySize > 0)
        self.config = config

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in NanoChatBlock(config) }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        hidden = functionalRMSNorm(hidden, eps: config.rmsNormEps)

        let mask = createAttentionMask(h: hidden, cache: cache)

        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }

        return functionalRMSNorm(hidden, eps: config.rmsNormEps)
    }
}

// MARK: - Public Model

public final class NanoChatModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let modelType: String

    let config: NanoChatConfiguration

    @ModuleInfo(key: "transformer") fileprivate var transformer: NanoChatModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: NanoChatConfiguration) {
        self.config = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)

        _transformer.wrappedValue = NanoChatModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = transformer(inputs, cache: cache)
        let logits = lmHead(hidden)
        return applySoftcap(logits, cap: config.logitsSoftcap)
    }
}

// MARK: - Configuration

public struct NanoChatConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var vocabularySize: Int
    public var maxPositionEmbeddings: Int
    public var intermediateSize: Int
    public var ropeTheta: Float
    public var rmsNormEps: Float
    public var logitsSoftcap: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case logitsSoftcap = "logits_softcap"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nanochat"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.maxPositionEmbeddings = try container.decode(
            Int.self, forKey: .maxPositionEmbeddings)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.logitsSoftcap =
            try container.decodeIfPresent(Float.self, forKey: .logitsSoftcap) ?? 15.0
    }
}

// MARK: - LoRA

extension NanoChatModel: LoRAModel {
    public var loraLayers: [Module] {
        transformer.layers
    }
}
