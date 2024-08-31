import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/phimoe.py

public struct PhiMoEConfiguration: Codable, Sendable {
    var modelType: String = "phimoe"
    var vocabularySize: Int = 32064
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 6400
    var hiddenLayers: Int = 32
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var maxPositionEmbeddings: Int = 131072
    var originalMaxPositionEmbeddings: Int = 4096
    var rmsNormEps: Float = 1e-6
    var ropeScaling: RopeScalingWithFactorArrays?
    var numLocalExperts: Int = 16
    var numExpertsPerToken: Int = 2
    var ropeTheta: Float = 10000.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case ropeTheta = "rope_theta"
    }
}

private class Attention: Module {
    let args: PhiMoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: SuScaledRotaryEmbedding

    init(_ args: PhiMoEConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: true)

        self.rope = SuScaledRotaryEmbedding(
            dimensions: headDim,
            base: args.ropeTheta,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            originalMaxPositionEmbeddings: args.originalMaxPositionEmbeddings,
            longFactor: args.ropeScaling?.longFactor as? [Float] ?? [1.0],
            longMScale: args.ropeScaling?.longMScale as? Float
        )
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        let queries = wq(x)
        let keys = wk(x)
        let values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        var q = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        var v = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
            (k, v) = cache.update(keys: k, values: v)
        } else {
            q = rope(q)
            k = rope(k)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class PhiMoESparseMoeBlock: Module {
    let hiddenDim: Int
    let ffnDim: Int
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: PhiMoEConfiguration) {
        self.hiddenDim = args.hiddenSize
        self.ffnDim = args.intermediateSize
        self.numExperts = args.numLocalExperts
        self.topK = args.numExpertsPerToken

        self._gate.wrappedValue = Linear(hiddenDim, numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: hiddenDim, hiddenDims: ffnDim, numExperts: numExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)

        let k = self.topK
        let inds = MLX.stopGradient(
            MLX.argPartition(
                -gates,
                kth: k - 1
                    // !! Here the Python has an extra argument that is not available in Swift: axis=-1
            )[.ellipsis, ..<k])
        let scores = MLX.softmax(MLX.take(gates, inds, axis: -1), axis: -1, precise: true)

        let y = switchMLP(x, inds)
        return (y * MLX.expandedDimensions(scores, axis: -1)).sum(axis: -2)  // !! Please check this line. In Python: (y * scores[..., None]).sum(axis=-2)
    }
}

private class PhiMoEDecoderLayer: Module {
    let hiddenSize: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: PhiMoESparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    init(_ args: PhiMoEConfiguration) {
        self.hiddenSize = args.hiddenSize

        self._selfAttn.wrappedValue = Attention(args)
        self._blockSparseMoe.wrappedValue = PhiMoESparseMoeBlock(args)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        var residual = x
        var hiddenStates = inputLayerNorm(x)
        hiddenStates = selfAttn(hiddenStates, mask: mask, cache: cache)
        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = blockSparseMoe(hiddenStates)
        hiddenStates = residual + hiddenStates

        return hiddenStates
    }
}

private class PhiMoEModelInner: Module {
    let args: PhiMoEConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [PhiMoEDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ args: PhiMoEConfiguration) {
        self.args = args

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in PhiMoEDecoderLayer(args) }
        self._norm.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class PhiMoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let headDim: IntOrPair

    fileprivate let model: PhiMoEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: PhiMoEConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.headDim = .init(args.hiddenSize / args.attentionHeads)
        self.model = PhiMoEModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: true)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< model.args.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for (n, m) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for k in ["weight", "scales", "biases"] {
                    if sanitizedWeights["\(prefix).block_sparse_moe.experts.0.\(n).\(k)"] != nil {
                        let toJoin = (0 ..< model.args.numLocalExperts).map { e in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).block_sparse_moe.experts.\(e).\(n).\(k)")!
                        }
                        sanitizedWeights["\(prefix).block_sparse_moe.switch_mlp.\(m).\(k)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension PhiMoEModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttn, ["q_proj", "v_proj"]) }
    }
}

// MARK: - SwitchGLU

class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let x = MLX.expandedDimensions(x, axes: [-2, -3])

        let xUp = upProj(x, indices)
        let xGate = gateProj(x, indices)
        let xDown = downProj(activation(xGate) * xUp, indices)

        return MLX.squeezed(xDown, axis: -2)
    }
}

class SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMatmul(x, weightT, rhsIndices: indices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(MLX.take(bias, indices), axis: -2)
        }

        return result
    }

    func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedSwitchLinear {
        let (numExperts, outputDims, inputDims) = (
            self.weight.shape[0], self.weight.shape[1], self.weight.shape[2]
        )
        let ql = QuantizedSwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            bias: self.bias != nil,
            groupSize: groupSize,
            bits: bits
        )

        let (quantizedWeight, scales, biases) = MLX.quantized(
            self.weight, groupSize: groupSize, bits: bits)
        ql.weight = quantizedWeight
        ql.scales = scales
        ql.biases = biases

        if let bias = self.bias {
            ql.bias = bias
        }

        return ql
    }
}

class QuantizedSwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let groupSize: Int
    let bits: Int

    init(
        inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true, groupSize: Int = 64,
        bits: Int = 4
    ) {
        self.groupSize = groupSize
        self.bits = bits

        let scale = sqrt(1.0 / Float(inputDims))
        let uniformWeight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        let (quantizedWeight, scales, biases) = MLX.quantized(
            uniformWeight, groupSize: groupSize, bits: bits)
        self._weight.wrappedValue = quantizedWeight
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
        self.freeze()
    }

    var inputDims: Int {
        return scales.shape[2] * groupSize
    }

    var outputDims: Int {
        return weight.shape[1]
    }

    var numExperts: Int {
        return weight.shape[0]
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var result = MLX.gatherQuantizedMatmul(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(MLX.take(bias, indices), axis: -2)
        }

        return result
    }
}
