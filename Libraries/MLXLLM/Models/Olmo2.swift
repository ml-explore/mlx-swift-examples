//  Olmo2.swift
//  LLM
//
//  Created by Sachin Desai on 9/11/25.
//

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/olmo2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - RoPE helpers

private class DynamicNTKScalingRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    var base: Float?
    let scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?
    var freqs: MLXArray?

    init(
        dims: Int,
        maxPositionEmbeddings: Int?,
        traditional: Bool = false,
        base: Float = 10000,
        scale: Float = 1.0,
        ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings ?? 2048
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
        super.init()
        computeFreqs()
    }

    private func computeFreqs() {
        if ropeType != "llama3" {
            freqs = nil
            return
        }

        guard let ropeScaling = ropeScaling,
            case .float(let factor) = ropeScaling["factor"],
            case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
            case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
            case .float(let oldContextLen) = ropeScaling["original_max_position_embeddings"]
                ?? .float(8192),
            let base
        else {
            freqs = nil
            return
        }

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen), frequencies * factor, frequencies)
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )
        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        self.base = nil
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset,
            freqs: freqs
        )
    }
}

// MARK: - Attention

private class Attention: Module {
    let args: Olmo2Configuration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    // rope can be either dynamic scaling or yarn depending on config
    let ropeDynamic: DynamicNTKScalingRoPE?
    let ropeYarn: YarnRoPE?

    init(_ args: Olmo2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: args.attentionBias)

        self._qNorm.wrappedValue = RMSNorm(dimensions: nHeads * headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: nKVHeads * headDim, eps: args.rmsNormEps)

        let ropeType: String = {
            if let v = args.ropeScaling?["type"] ?? args.ropeScaling?["rope_type"],
                case .string(let s) = v
            {
                return s
            } else {
                return "default"
            }
        }()

        if ropeType == "yarn" {
            // Map the Yarn parameters with sensible defaults
            let factor = args.ropeScaling?["factor"]?.asFloat() ?? 32.0
            let origMax = args.ropeScaling?["original_max_position_embeddings"]?.asInt() ?? 4096
            let betaFast = args.ropeScaling?["beta_fast"]?.asFloat() ?? 32.0
            let betaSlow = args.ropeScaling?["beta_slow"]?.asFloat() ?? 1.0
            let mscale = args.ropeScaling?["mscale"]?.asFloat() ?? 1.0
            let mscaleAllDim = args.ropeScaling?["mscale_all_dim"]?.asFloat() ?? 0.0
            self.ropeYarn = YarnRoPE(
                dimensions: headDim,
                traditional: args.ropeTraditional,
                maxPositionEmbeddings: args.maxPositionEmbeddings ?? 2048,
                base: args.ropeTheta,
                scalingFactor: factor,
                originalMaxPositionEmbeddings: origMax,
                betaFast: betaFast,
                betaSlow: betaSlow,
                mscale: mscale,
                mscaleAllDim: mscaleAllDim
            )
            self.ropeDynamic = nil
        } else {
            // Support default/linear/llama3
            let ropeScale: Float
            if ropeType == "linear", let factor = args.ropeScaling?["factor"]?.asFloat() {
                ropeScale = 1 / factor
            } else {
                ropeScale = 1
            }
            self.ropeDynamic = DynamicNTKScalingRoPE(
                dims: headDim,
                maxPositionEmbeddings: args.maxPositionEmbeddings,
                traditional: args.ropeTraditional,
                base: args.ropeTheta,
                scale: ropeScale,
                ropeType: ropeType,
                ropeScaling: args.ropeScaling
            )
            self.ropeYarn = nil
        }
    }

    func applyRoPE(_ x: MLXArray, offset: Int?) -> MLXArray {
        if let ropeYarn { return ropeYarn(x, offset: offset ?? 0) }
        if let ropeDynamic {
            return ropeDynamic(x, offset: offset ?? 0)
        }
        return x
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x))
        var keys = kNorm(wk(x))
        var values = wv(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = applyRoPE(queries, offset: cache.offset)
            keys = applyRoPE(keys, offset: cache.offset)
        } else {
            queries = applyRoPE(queries, offset: nil)
            keys = applyRoPE(keys, offset: nil)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: Olmo2Configuration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP

    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    init(_ args: Olmo2Configuration) {
        self._attention.wrappedValue = Attention(args)
        self._mlp.wrappedValue = MLP(args)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = postAttentionLayerNorm(attention(x, mask: mask, cache: cache))
        let h = x + r
        r = postFeedforwardLayerNorm(mlp(h))
        let out = h + r
        return out
    }
}

// MARK: - Model

private class Olmo2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: Olmo2Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class Olmo2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: Olmo2ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Olmo2Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Olmo2ModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }
}

// MARK: - Configuration

public struct Olmo2Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    var resolvedHeadDimensions: Int { headDimensions ?? (hiddenSize / attentionHeads) }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        let maybeKV = try container.decodeIfPresent(Int.self, forKey: .kvHeads)
        kvHeads = maybeKV ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
        {
            self.ropeTraditional = ropeTraditional
        }
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }
    }
}

// MARK: - LoRA

extension Olmo2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
