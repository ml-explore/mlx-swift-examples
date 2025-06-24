// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable
import Tokenizers

// Port of https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models/gemma2.py

private class Attention: Module {
    let args: Gemma2Configuration
    let scale: Float
    let logitSoftCap: Float
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: Gemma2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.repeats = args.attentionHeads / args.kvHeads
        self.headDim = args.headDimensions

        self.scale = 1.0 / pow(Float(args.queryPreAttnScalar), 0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        self.logitSoftCap = args.attnLogitSoftcapping
        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        queries = queries * self.scale

        if repeats > 1 {
            queries = queries.reshaped([B, nKVHeads, repeats, L, headDim])
            keys = expandedDimensions(keys, axes: [2])
            values = expandedDimensions(values, axes: [2])
        }

        var scores = matmul(queries, keys.swappedAxes(-1, -2))
        scores = tanh(scores / logitSoftCap) * logitSoftCap

        if let mask {
            scores = scores + mask
        }
        scores = softmax(scores, axis: -1, precise: true)
        var output = matmul(scores, values)
        if repeats > 1 {
            output = output.reshaped([B, nHeads, L, headDim])
        }
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(gelu(gate(x)) * up(x))
    }
}

// Minimal changes from Gemma TransformerBlock
private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

    public init(_ args: Gemma2Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(preFeedforwardLayerNorm(h))
        let out = h + postFeedforwardLayerNorm(r)
        return out
    }
}

// Uses Gemma2TransformerBlock, otherwise same as GemmaModelInner
private class ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    fileprivate let norm: Gemma.RMSNorm

    let hiddenScale: Float

    public init(_ args: Gemma2Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.hiddenScale = pow(Float(args.hiddenSize), 0.5)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
        self.norm = Gemma.RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * hiddenScale

        let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// Uses Gemma2ModelInner, otherwise same as GemmaModel
public class Gemma2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: ModelInner
    let logitSoftCap: Float

    public init(_ args: Gemma2Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = ModelInner(args)
        self.logitSoftCap = args.finalLogitSoftcapping
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = model.embedTokens.asLinear(out)
        out = tanh(out / logitSoftCap) * logitSoftCap
        return out
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        NoSystemMessageGenerator()
    }
}

@Codable
public struct Gemma2Configuration: Sendable {
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("head_dim") public var headDimensions: Int
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int
    @CodingKey("rope_theta") public var ropeTheta: Float = 10_000
    @CodingKey("rope_traditional") public var ropeTraditional: Bool = false
    @CodingKey("attn_logit_softcapping") public var attnLogitSoftcapping: Float = 50.0
    @CodingKey("final_logit_softcapping") public var finalLogitSoftcapping: Float = 30.0
    @CodingKey("query_pre_attn_scalar") public var queryPreAttnScalar: Float = 144.0
}

// MARK: - LoRA

extension Gemma2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
