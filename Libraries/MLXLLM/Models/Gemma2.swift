// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma2.py

// Specialized norm for gemma
private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}

private class Attention: Module {
    let args: Gemma2Configuration
    let scale: Float
    let logitSoftCap: Float
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int

    @ModuleInfo(key: "q_proj") var wq: UnaryLayer
    @ModuleInfo(key: "k_proj") var wk: UnaryLayer
    @ModuleInfo(key: "v_proj") var wv: UnaryLayer
    @ModuleInfo(key: "o_proj") var wo: UnaryLayer

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
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
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

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Gemma2Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(preFeedforwardLayerNorm(h))
        let out = h + postFeedforwardLayerNorm(r)
        return out
    }
}

// Uses Gemma2TransformerBlock, otherwise same as GemmaModelInner
public class ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    fileprivate let norm: RMSNorm

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
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

    let model: ModelInner
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
}

public struct Gemma2Configuration: Codable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var attnLogitSoftcapping: Float = 50.0
    var finalLogitSoftcapping: Float = 30.0
    var queryPreAttnScalar: Float = 144.0

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case attnLogitSoftcapping = "attn_logit_softcapping"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case queryPreAttnScalar = "query_pre_attn_scalar"
    }

    public init(from decoder: Decoder) throws {
        // Custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<CodingKeys> = try decoder.container(
            keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: CodingKeys.attentionHeads)
        self.headDimensions = try container.decode(
            Int.self, forKey: CodingKeys.headDimensions)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: CodingKeys.ropeTheta)
            ?? 10_000
        self.ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        self.attnLogitSoftcapping = try container.decode(
            Float.self, forKey: CodingKeys.attnLogitSoftcapping)
        self.finalLogitSoftcapping = try container.decode(
            Float.self, forKey: CodingKeys.finalLogitSoftcapping)
        self.queryPreAttnScalar = try container.decode(
            Float.self, forKey: CodingKeys.queryPreAttnScalar)
    }
}

// MARK: - LoRA

extension Gemma2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}

// TODO - notes
//
// - make UnaryLayer extend Module
// - make a Quantized protocol that provides the groupSize and bits
// - make the QuantizedLinear shape produce the expanded shape
// - make `items()` open
// - make `updateModule(key:_:)` open
// - make `noGrad` overridable (turn into function?)
//
// - evaluation and training should work as expected
// - this flattens the weights and modules into one layer
//      - to match the normal lora implementation
//      - see items() and updateModule()

// TODO: make UnaryLayer extend Module
public protocol UnaryLayer2: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

/// LoRA layer that can wrap any UnaryLayer
class LoRA: Module, UnaryLayer2 {

    let adapts: UnaryLayer2
    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    public init(
        adapts: UnaryLayer2, inputDimensions: Int, outputDimensions: Int, rank: Int = 8,
        scale: Float = 20.0
    ) {
        self.adapts = adapts

        self.scale = scale

        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        freeze()
    }

    // TODO: in LoRALinear this is
    // public static func from(linear: Linear, rank: Int = 8) -> LoRA
    public convenience init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
        var (outputDimensions, inputDimensions) = linear.shape

        if let linear = linear as? QuantizedLinear {
            // TODO Linear should probably have a property to return these directly
            // rather than shape which represents the physical shape of the layers
            inputDimensions = inputDimensions * 32 / linear.bits
        }

        self.init(
            adapts: linear,
            inputDimensions: inputDimensions, outputDimensions: outputDimensions,
            rank: rank, scale: scale)
    }

    // produce a merged view of properties (flatten LoRA into adapts)
    override func items() -> ModuleItems {
        var result = adapts.items()
        for (key, value) in super.items() {
            if key == "adapts" { continue }
            result[key] = value
        }
        return result
    }

    // forward module updates -> adapt
    func updateModule(key: String, _ value: Any) throws {
        try adapts.updateModule(key: key, value)
    }

    override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws
    {
        try adapts.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    // TODO: this requires knowledge of the innards of the adapted layer so it
    // is specific to Linear (and QuantizedLinear).
    public func toLinear(deQuantize: Bool = false) -> Linear {
        // TODO throws?  failable?
        guard let linear = adapts as? Linear else { fatalError("Not a Linear") }

        var weight: MLXArray
        if let quantized = linear as? QuantizedLinear {
            weight = dequantized(
                quantized.weight, scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits)
        } else {
            weight = linear.weight
        }

        let loraB = (scale * loraB.T).asType(.float16)
        let loraA = loraA.T.asType(.float16)
        let mergedWeight = weight + matmul(loraB, loraA)

        // TODO maybe add a protocol for Quanitzed
        if let quantized = linear as? QuantizedLinear {
            return QuantizedLinear(
                weight: mergedWeight, bias: quantized.bias,
                groupSize: quantized.groupSize, bits: quantized.bits)
        } else {
            return Linear(weight: mergedWeight, bias: linear.bias)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO let y = super.callAsFunction(x.asType(scales.dtype)) -- ignoring the asType here
        let y = adapts(x)
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}
