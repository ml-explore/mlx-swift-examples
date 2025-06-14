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

// MARK: - BitLinear Layer

private func compileMatmulKernel() -> MLXFastKernel {
    let source = """
        uint tid = thread_position_in_grid.x;
        uint total_elements = batch_size * out_features;

        if (tid >= total_elements) return;

        uint batch_idx = tid / out_features;
        uint out_idx = tid % out_features;

        float sum = 0.0;

        // Calculate packed dimensions
        uint packed_rows = out_features / 4;  // Each packed row contains 4 output rows

        for (uint i = 0; i < in_features; i++) {
            // Get input value
            float x_val = x[batch_idx * in_features + i];

            // Determine which packed row and which bit position within that packed value
            uint which_slice = out_idx / packed_rows;  // Which of the 4 slices (0, 1, 2, 3)
            uint row_in_slice = out_idx - which_slice * packed_rows;  // Which row within that slice

            // Get the packed weight value
            uint packed_idx = row_in_slice * in_features + i;
            uint8_t packed_val = packed_weights[packed_idx];


            // Extract the 2-bit slice; {0,1,2} -> {-1,0,1} (11 is unused and would map to 2)
            float weight_val = float((packed_val >> (2 * which_slice)) & 3) - 1.0;

            sum += x_val * weight_val;
        }

        // Apply weight scaling by diving them or multiplying them
        out[tid] = invert_weight_scales ? (sum / weight_scale[0]) : (sum * weight_scale[0]);
        """

    return metalKernel(
        name: "bitlinear_matmul",
        inputNames: ["x", "packed_weights", "weight_scale", "invert_weight_scales"],
        outputNames: ["out"],
        source: source
    )
}

private class BitLinear: Module {
    let inFeatures: Int
    let outFeatures: Int
    let dtype: DType
    let invertWeightScales: Bool

    let weight: MLXArray
    let bias: MLXArray?
    @ModuleInfo(key: "weight_scale") var weightScale: MLXArray

    private let compiledKernel: MLXFast.MLXFastKernel

    init(
        _ inFeatures: Int,
        _ outFeatures: Int,
        bias: Bool = true,
        dtype: DType = .float16,
        invertWeightScales: Bool = false
    ) {
        self.dtype = dtype
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures

        let packedOutFeatures = Int(floor(Double(outFeatures + 3) / 4.0))
        self.weight = MLXArray.zeros([packedOutFeatures, inFeatures], dtype: .uint8)

        self.invertWeightScales = invertWeightScales
        self._weightScale.wrappedValue = MLXArray([1]).asType(dtype)

        if bias {
            self.bias = MLXArray.zeros([outFeatures], dtype: dtype)
        } else {
            self.bias = nil
        }

        self.compiledKernel = compileMatmulKernel()

        super.init()
    }

    private func executeMatmulKernel(_ x: MLXArray, _ packedWeights: MLXArray) -> MLXArray {
        let originalShape = x.shape
        let xFlattened: MLXArray
        let totalBatchElements: Int
        let inFeatures: Int

        if originalShape.count > 2 {
            xFlattened = x.reshaped(-1, originalShape[originalShape.count - 1])
            totalBatchElements = xFlattened.dim(0)
            inFeatures = xFlattened.dim(1)
        } else {
            xFlattened = x
            totalBatchElements = xFlattened.dim(0)
            inFeatures = xFlattened.dim(1)
        }

        let outFeatures = outFeatures

        let outputs = compiledKernel(
            [xFlattened.asType(dtype), packedWeights, weightScale, MLXArray(invertWeightScales)],
            template: [
                ("batch_size", totalBatchElements),
                ("in_features", inFeatures),
                ("out_features", outFeatures),
            ],
            grid: (totalBatchElements * outFeatures, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalBatchElements, outFeatures]],
            outputDTypes: [dtype]
        )

        if originalShape.count > 2 {
            let outputShape = Array(originalShape.dropLast()) + [outFeatures]
            return outputs[0].reshaped(outputShape)
        } else {
            return outputs[0]
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalDtype = x.dtype

        var y = executeMatmulKernel(x, weight)

        if let bias {
            y = y + bias
        }
        return y.asType(originalDtype)
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

// MARK: - ReLU²

private func reluSquared(_ x: MLXArray) -> MLXArray {
    compile(shapeless: true) {
        relu($0).square()
    }(x)
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
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
