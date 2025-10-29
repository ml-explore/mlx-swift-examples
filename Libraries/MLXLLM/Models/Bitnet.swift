//
//  Bitnet.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/6/12.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bitnet.py

private func makeBitLinearKernel() -> MLXFast.MLXFastKernel {
    let source = """
        constexpr int M = 4;
        constexpr int BLOCK = 32;

        uint tid = thread_position_in_grid.y;
        uint in_offset = thread_position_in_grid.x;

        uint batch_idx = tid / (out_features / 4);
        uint row_idx = tid % (out_features / 4);

        float sum[4] = {0.0};

        for (uint i = in_offset * M; i < in_features; i += BLOCK * M) {
            float v[M];
            for (int j=0; j<M; j++) {
                v[j] = x[batch_idx * in_features + i + j];
            }

            for (int j=0; j<M; j++) {
                uint8_t w = packed_weights[row_idx * in_features + i + j];
                sum[0] += v[j] * ((w & 3) - 1);
                sum[1] += v[j] * (((w >> 2) & 3) - 1);
                sum[2] += v[j] * (((w >> 4) & 3) - 1);
                sum[3] += v[j] * (((w >> 6) & 3) - 1);
            }
        }

        for (int j=0; j<4; j++) {
            sum[j] = simd_sum(sum[j]);
        }

        // Apply weight scaling by diving them or multiplying them
        if (in_offset == 0) {
            float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
            for (int i=0; i<4; i++) {
                out[batch_idx * out_features + row_idx + i * (out_features/4)] = static_cast<T>(sum[i] * scale);
            }
        }
        """

    return metalKernel(
        name: "bitlinear_matmul",
        inputNames: ["x", "packed_weights", "weight_scale"],
        outputNames: ["out"],
        source: source
    )
}

private final class BitLinearKernelManager: @unchecked Sendable {
    static let shared = BitLinearKernelManager()

    let bitlinearKernel: MLXFast.MLXFastKernel

    private init() {
        bitlinearKernel = makeBitLinearKernel()
    }
}

private class BitLinear: Module {
    let inFeatures: Int
    let outFeatures: Int
    let invertWeightScales: Bool

    let weight: MLXArray
    let bias: MLXArray?
    @ModuleInfo(key: "weight_scale") var weightScale: MLXArray

    init(
        _ inFeatures: Int,
        _ outFeatures: Int,
        bias: Bool = true,
        invertWeightScales: Bool = false
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures

        let packedOutFeatures = Int(floor(Double(outFeatures + 3) / 4.0))
        self.weight = MLXArray.zeros([packedOutFeatures, inFeatures], dtype: .uint8)

        self.invertWeightScales = invertWeightScales
        self._weightScale.wrappedValue = MLXArray([1.0])

        if bias {
            self.bias = MLXArray.zeros([outFeatures])
        } else {
            self.bias = nil
        }

        super.init()
    }

    private func executeMatmulKernel(_ x: MLXArray, _ packedWeights: MLXArray) -> MLXArray {
        let originalShape = x.shape
        var x = x

        if originalShape.count > 2 {
            x = x.reshaped(-1, originalShape[originalShape.count - 1])
        }

        let totalBatchElements = x.dim(0)
        let inFeatures = x.dim(1)

        let outFeatures = self.outFeatures

        let dtype = self.weightScale.dtype
        assert(x.dtype == dtype, "Wrong type for input.")

        var outputs = BitLinearKernelManager.shared.bitlinearKernel(
            [x, packedWeights, weightScale],
            template: [
                ("T", dtype),
                ("invert_weight_scales", invertWeightScales),
                ("in_features", inFeatures),
                ("out_features", outFeatures),
            ],
            grid: (32, Int(floor(Double(totalBatchElements * outFeatures / 4))), 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalBatchElements, outFeatures]],
            outputDTypes: [dtype]
        )[0]

        if originalShape.count > 2 {
            outputs = outputs.reshaped(Array(originalShape.dropLast()) + [outFeatures])
        }

        return outputs
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = executeMatmulKernel(x, weight)

        if let bias {
            y = y + bias
        }
        return y
    }
}

// MARK: - Model Configuration

public struct BitnetConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var kvHeads: Int?
    var attentionBias: Bool
    var mlpBias: Bool
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    public init(
        modelType: String = "bitnet",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        headDimensions: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        kvHeads: Int? = nil,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        ropeTheta: Float = 10000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDimensions = headDimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    var resolvedKvHeads: Int {
        kvHeads ?? attentionHeads
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "bitnet"
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings
        )
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling
        )
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

// MARK: - Attention

private class Attention: Module {
    let args: BitnetConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: BitLinear
    @ModuleInfo(key: "k_proj") var kProj: BitLinear
    @ModuleInfo(key: "v_proj") var vProj: BitLinear
    @ModuleInfo(key: "o_proj") var oProj: BitLinear

    @ModuleInfo(key: "attn_sub_norm") var attnSubNorm: RMSNorm

    let rope: RoPE

    init(_ args: BitnetConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let headDim = args.resolvedHeadDimensions
        let nHeads = args.attentionHeads
        let nKvHeads = args.resolvedKvHeads

        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = BitLinear(dim, nHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = BitLinear(dim, nKvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = BitLinear(dim, nKvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = BitLinear(nHeads * headDim, dim, bias: args.attentionBias)

        _attnSubNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta,
            scale: ropeScale
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.resolvedKvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.resolvedKvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        let normedOutput = attnSubNorm(output)
        return oProj(normedOutput)
    }
}

// MARK: - MLP

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: BitLinear
    @ModuleInfo(key: "down_proj") var downProj: BitLinear
    @ModuleInfo(key: "up_proj") var upProj: BitLinear

    @ModuleInfo(key: "ffn_sub_norm") var ffnSubNorm: RMSNorm

    init(_ args: BitnetConfiguration) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize

        _gateProj.wrappedValue = BitLinear(dim, hiddenDim, bias: args.mlpBias)
        _downProj.wrappedValue = BitLinear(hiddenDim, dim, bias: args.mlpBias)
        _upProj.wrappedValue = BitLinear(dim, hiddenDim, bias: args.mlpBias)
        _ffnSubNorm.wrappedValue = RMSNorm(dimensions: args.intermediateSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = reluSquared(gateProj(x)) * upProj(x)
        let normed = ffnSubNorm(gated)
        return downProj(normed)
    }
}

// MARK: - Transformer Block

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    var mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: BitnetConfiguration) {
        _attention.wrappedValue = Attention(args)
        mlp = MLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

// MARK: - Bitnet Model Inner

private class BitnetModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    var norm: RMSNorm

    init(_ args: BitnetConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize
        )

        layers = (0 ..< args.hiddenLayers).map { _ in
            TransformerBlock(args)
        }
        norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

// MARK: - Bitnet Model

public class BitnetModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: BitnetModelInner
    let configuration: BitnetConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: BitnetConfiguration) {
        configuration = args
        vocabularySize = args.vocabularySize
        kvHeads = (0 ..< args.hiddenLayers).map { _ in args.resolvedKvHeads }
        model = BitnetModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
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
        var weights = weights

        weights = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

extension BitnetModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
