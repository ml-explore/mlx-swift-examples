//
//  GPTOSS.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/8/6.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

// MARK: - Configuration

public struct GPTOSSConfiguration: Codable, Sendable {
    public var modelType: String = "gpt_oss"
    public var hiddenLayers: Int = 36
    public var localExperts: Int = 128
    public var expertsPerToken: Int = 4
    public var vocabularySize: Int = 201088
    public var rmsNormEps: Float = 1e-5
    public var hiddenSize: Int = 2880
    public var intermediateSize: Int = 2880
    public var headDim: Int = 64
    public var attentionHeads: Int = 64
    public var kvHeads: Int = 8
    public var slidingWindow: Int = 128
    public var ropeTheta: Float = 150000
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var layerTypes: [String]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.localExperts = try container.decode(Int.self, forKey: .localExperts)
        self.expertsPerToken = try container.decode(Int.self, forKey: .expertsPerToken)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 150000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
    }
}

private func mlxTopK(_ a: MLXArray, k: Int, axis: Int = -1) -> (values: MLXArray, indices: MLXArray)
{
    let partitionedIndices = argPartition(a, kth: -k, axis: axis)
    let topKIndices = partitionedIndices[.ellipsis, (-k)...]
    let topKValues = takeAlong(a, topKIndices, axis: axis)
    return (topKValues, topKIndices)
}

private func swiglu(_ xLinear: MLXArray, _ xGlu: MLXArray, alpha: Float = 1.702, limit: Float = 7.0)
    -> MLXArray
{
    var xLinear = xLinear
    var xGlu = xGlu
    xGlu = clip(xGlu, max: MLXArray(limit))
    xLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

    let gluScaled = alpha * xGlu
    let sig = sigmoid(gluScaled)

    let outGlu = xGlu * sig
    return outGlu * (xLinear + 1)
}

private func compileSwiglu() -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    compile(shapeless: true) { xLinear, xGlu in
        swiglu(xLinear, xGlu)
    }
}

class SwiGLUSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size > 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            compileSwiglu()(xUp, xGate),
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return x.squeezed(axis: -2)
    }
}

private class AttentionBlock: Module {
    let headDim: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numKeyValueGroups: Int
    let smScale: Float

    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: YarnRoPE

    private var _previousMask: MLXArray?

    public init(_ config: GPTOSSConfiguration) {
        self.headDim = config.headDim
        self.numAttentionHeads = config.attentionHeads
        self.numKeyValueHeads = config.kvHeads
        self.numKeyValueGroups = config.attentionHeads / config.kvHeads

        _sinks.wrappedValue = zeros([config.attentionHeads])
        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: true)
        _oProj.wrappedValue = Linear(
            config.headDim * config.attentionHeads, config.hiddenSize, bias: true)

        self.smScale = 1.0 / sqrt(Float(config.headDim))

        if let ropeScaling = config.ropeScaling {
            self.rope = YarnRoPE(
                dimensions: headDim,
                base: config.ropeTheta,
                scalingFactor: ropeScaling["factor"]?.asFloat() ?? 32.0,
                originalMaxPositionEmbeddings: ropeScaling["original_max_position_embeddings"]?
                    .asInt() ?? 4096,
                betaFast: ropeScaling["beta_fast"]?.asFloat() ?? 32.0,
                betaSlow: ropeScaling["beta_slow"]?.asFloat() ?? 1.0
            )
        } else {
            self.rope = YarnRoPE(
                dimensions: headDim,
                base: config.ropeTheta
            )
        }
    }

    func getCausalMask(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let L = x.dim(1)
        var offset = cache?.offset ?? 0
        offset = max(1, offset)

        func makeMask(_ L: Int, _ offset: Int) -> MLXArray {
            let zero = MLXArray(0, dtype: x.dtype)
            let neginf = MLXArray(-Float.infinity, dtype: x.dtype)
            var mask = MLX.where(createCausalMask(n: L, offset: offset - 1), zero, neginf)
            mask = mask.reshaped(1, 1, L, -1)
            mask = tiled(mask, repetitions: [1, numAttentionHeads, 1, 1])
            let sinks = tiled(sinks.reshaped(1, -1, 1, 1), repetitions: [1, 1, L, 1])
            mask = concatenated([sinks, mask], axis: -1)
            return mask
        }

        if L > 8 {
            _previousMask = nil
            return makeMask(L, offset)
        }

        let length = ((L + offset + 511) / 512) * 512
        if _previousMask == nil || _previousMask!.dim(-1) < length
            || _previousMask!.dim(-2) != L
        {
            _previousMask = makeMask(L, length - L)
        }

        return _previousMask![.ellipsis, 0 ..< L + offset]
    }

    func getSlidingWindowMask(_ x: MLXArray, cache: KVCache?, windowSize: Int) -> MLXArray {
        let L = x.dim(1)
        var offset = cache?.offset ?? 0
        offset = max(1, offset)

        func makeMask(_ L: Int, _ offset: Int) -> MLXArray {
            let zero = MLXArray(0, dtype: x.dtype)
            let neginf = MLXArray(-Float.infinity, dtype: x.dtype)
            var mask = createCausalMask(n: L, offset: offset - 1, windowSize: windowSize)
            mask = MLX.where(mask, zero, neginf)
            mask = mask.reshaped(1, 1, L, -1)
            mask = tiled(mask, repetitions: [1, numAttentionHeads, 1, 1])
            let sinks = tiled(sinks.reshaped(1, -1, 1, 1), repetitions: [1, 1, L, 1])
            mask = concatenated([sinks, mask], axis: -1)
            return mask
        }

        if L > 1 {
            _previousMask = nil
            return makeMask(L, min(windowSize + 1, offset))
        }

        if _previousMask == nil {
            _previousMask = makeMask(L, windowSize + 1)
        }

        return _previousMask![.ellipsis, 0 ..< min(L + offset, windowSize + 1)]
    }

    func getMask(_ x: MLXArray, cache: KVCache?, windowSize: Int?) -> MLXArray {
        if let windowSize {
            return getSlidingWindowMask(x, cache: cache, windowSize: windowSize)
        } else {
            return getCausalMask(x, cache: cache)
        }
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let D = headDim
        let Hk = numKeyValueHeads

        var q = qProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        var k = kProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        var v = vProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)

        // Quantized cache path
        if let qcache = cache as? QuantizedKVCacheProtocol {
            if qcache.offset == 0 {
                q = rope(q)
                k = rope(k)

                let zeros = MLXArray.zeros([B, Hk, 1, D]).asType(k.dtype)
                k = concatenated([zeros, k], axis: 2)
                v = concatenated([zeros, v], axis: 2)
            } else {
                q = rope(q, offset: qcache.offset - 1)
                k = rope(k, offset: qcache.offset - 1)
            }

            let (qKeys, qValues) = qcache.updateQuantized(keys: k, values: v)
            let vHat = quantizedScaledDotProductAttention(
                queries: q,
                quantizedKeys: qKeys,
                quantizedValues: qValues,
                scale: smScale,
                mask: .array(mask),
                groupSize: qcache.groupSize,
                bits: qcache.bits,
                mode: qcache.mode
            )

            return oProj(vHat.swappedAxes(1, 2).reshaped(B, L, -1))
        }

        if cache == nil || cache?.offset == 0 {
            q = rope(q)
            k = rope(k)

            let zeros = MLXArray.zeros([B, Hk, 1, D]).asType(k.dtype)
            k = concatenated([zeros, k], axis: 2)
            v = concatenated([zeros, v], axis: 2)
            if let cache {
                (k, v) = cache.update(keys: k, values: v)
            }
        } else {
            q = rope(q, offset: cache!.offset - 1)
            k = rope(k, offset: cache!.offset - 1)
            (k, v) = cache!.update(keys: k, values: v)
        }

        let vHat = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: smScale,
            mask: mask)

        return oProj(vHat.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

private class MLPBlock: Module {
    let hiddenSize: Int
    let numLocalExperts: Int
    let numExpertsPerTok: Int

    @ModuleInfo(key: "experts") var experts: SwiGLUSwitchGLU
    @ModuleInfo(key: "router") var router: Linear

    public init(_ config: GPTOSSConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.numLocalExperts = config.localExperts
        self.numExpertsPerTok = config.expertsPerToken

        _experts.wrappedValue = SwiGLUSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.localExperts,
            bias: true
        )
        _router.wrappedValue = Linear(config.hiddenSize, config.localExperts, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = router(x)
        let (experts, indices) = mlxTopK(g, k: numExpertsPerTok, axis: -1)
        let expertWeights = softmax(experts, axis: -1, precise: true)

        var x = self.experts(x, indices)

        x = x * expandedDimensions(expertWeights, axis: -1)
        return x.sum(axis: -2)
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AttentionBlock
    @ModuleInfo(key: "mlp") var mlp: MLPBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ config: GPTOSSConfiguration) {
        _selfAttn.wrappedValue = AttentionBlock(config)
        _mlp.wrappedValue = MLPBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray, cache: KVCache? = nil) -> MLXArray {
        var residual = x
        var x = inputLayerNorm(x)
        x = selfAttn(x, mask: mask, cache: cache)
        x = residual + x

        residual = x
        x = postAttentionLayerNorm(x)
        x = mlp(x)
        x = residual + x
        return x
    }
}

private class ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layerTypes: [String]
    fileprivate let layers: [TransformerBlock]
    let windowSize: Int

    public init(_ config: GPTOSSConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layerTypes =
            config.layerTypes
            ?? Array(
                repeating: [
                    "sliding_attention",
                    "full_attention",
                ], count: config.hiddenLayers / 2
            ).flatMap { $0 }
        self.layers = (0 ..< config.hiddenLayers).map { _ in TransformerBlock(config) }
        self.windowSize = config.slidingWindow
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var x: MLXArray
        if let inputEmbeddings {
            x = inputEmbeddings
        } else {
            x = embedTokens(inputs)
        }

        let cache: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)

        var masks: [MLXArray]
        if let mask {
            masks = [MLXArray](repeating: mask, count: layers.count)
        } else {
            masks = []
            for (i, layer) in layers.enumerated() {
                masks.append(
                    layer.selfAttn.getMask(
                        x,
                        cache: cache[i],
                        windowSize: layerTypes[i] == "sliding_attention" ? windowSize : nil
                    )
                )
            }
        }

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: masks[i], cache: cache[i])
        }

        x = norm(x)

        return x
    }
}

private func convertMoePackedTensors(blocks: MLXArray, scales: MLXArray) -> MLXArray {
    precondition(
        blocks.shape.dropLast() == scales.shape,
        "blocks.shape=\(blocks.shape) does not match scales.shape=\(scales.shape)"
    )

    var scales = scales.asType(.int32) - 127
    let lut = MLXArray([
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]).asType(.bfloat16)

    let (prefixShape, G, B) = (Array(blocks.shape.dropLast(2)), blocks.dim(-2), blocks.dim(-1))

    let blocks = blocks.reshaped(-1, B)
    scales = scales.reshaped(-1, 1)

    let idxLo = blocks & 0x0F
    let idxHi = blocks >> 4

    var out = stacked([lut[idxLo], lut[idxHi]], axis: -1).flattened(start: -2)
    out = (2.0 ** scales) * out
    out = out.reshaped(prefixShape.count, G * B * 2)
    return out.asType(.bfloat16)
}

public class GPTOSSModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]
    private let model: ModelInner
    private let configuration: GPTOSSConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: GPTOSSConfiguration) {
        self.configuration = config
        self.modelType = config.modelType
        self.model = ModelInner(config)
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if weights.keys.contains(where: { $0.contains("gate_proj.weight") }) {
            return weights
        }

        if weights.keys.contains(where: { $0.contains("gate_up_proj_scales") }) {
            var newWeights: [String: MLXArray] = [:]
            for (k, v) in weights {
                if k.hasSuffix("_scales") {
                    continue
                } else if k.hasSuffix("_blocks") {
                    let scaleKey = k.replacingOccurrences(of: "_blocks", with: "_scales")
                    if let scales = weights[scaleKey] {
                        let newV = convertMoePackedTensors(blocks: v, scales: scales)
                        let newK = k.replacingOccurrences(of: "_blocks", with: "")
                        newWeights[newK] = newV
                    }
                } else {
                    newWeights[k] = v
                }
            }
            weights = newWeights
        }

        var finalWeights: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("gate_up_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "gate_proj.weight")
                ] = v[.ellipsis, .stride(by: 2), 0...]
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "up_proj.weight")
                ] = v[.ellipsis, .stride(from: 1, by: 2), 0...]
            } else if k.contains("down_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj", with: "down_proj.weight")
                ] = v
            } else if k.contains("gate_up_proj_bias") {
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "gate_proj.bias")
                ] = v[.ellipsis, .stride(by: 2)]
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "up_proj.bias")
                ] = v[.ellipsis, .stride(from: 1, by: 2)]
            } else if k.contains("down_proj_bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj_bias", with: "down_proj.bias")
                ] = v
            } else {
                finalWeights[k] = v
            }
        }

        return finalWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [KVCache] = []

        for lt in model.layerTypes {
            if lt == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(
                    RotatingKVCache(maxSize: configuration.slidingWindow + 1, keep: 1)
                )
            }
        }

        return caches
    }
}

extension GPTOSSModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
