//
//  Gemma3Text.swift
//  mlx-swift-examples
//
//  Created by Anthony DePasquale on 14.03.2025.
//

// Based on https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma3_text.py

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

/// Create a bidirectional sliding window mask where tokens can attend to others within the sliding window distance
public func createBidirectionalSlidingWindowMask(
    n: Int,
    offset: Int,
    windowSize: Int
) -> MLXArray {
    let rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    let rindsBcast = rinds[.newAxis]
    
    // Create mask where abs(q_idx - kv_idx) < windowSize (bidirectional window)
    let distance = abs(linds - rindsBcast)
    let mask = distance .< windowSize
    
    return mask
}

func simpleSDPA(queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray, scale: Float) -> MLXArray {
    var attn = matmul(queries, keys.transposed(0, 1, 3, 2))
    attn = attn - (1 - mask) * 1e6
    let weights = softmax(scale * attn, axis:-1)
    return matmul(weights, values)
}

public struct Gemma3TextConfiguration: Codable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let kvHeads: Int
    public let ropeGlobalBaseFreq: Float
    public let ropeLocalBaseFreq: Float
    public let ropeTraditional: Bool
    public let queryPreAttnScalar: Float
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let useBidirectionalAttention: Bool
    public let quantizationConfig: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case quantizationConfig = "quantization"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int, ropeGlobalBaseFreq: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool, queryPreAttnScalar: Float, slidingWindow: Int, slidingWindowPattern: Int, useBidirectionalAttention: Bool, quantizationConfig: QuantizationConfig? = nil) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeGlobalBaseFreq = ropeGlobalBaseFreq
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.useBidirectionalAttention = useBidirectionalAttention
        self.quantizationConfig = quantizationConfig
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // in the case of VLM models convertered using mlx_lm.convert
        // the configuration will still match the VLMs and be under text_config
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        useBidirectionalAttention = 
            try container.decodeIfPresent(Bool.self, forKey: .useBidirectionalAttention) ?? false
        
        let rawSlidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        // Apply sliding window adjustment for bidirectional attention (from patch: (sliding_window // 2) + 1)
        slidingWindow = useBidirectionalAttention ? (rawSlidingWindow / 2) + 1 : rawSlidingWindow
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6

        quantizationConfig = try container.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig)
    }
}

// MARK: - Quantization Configuration

public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let useBidirectionalAttention: Bool

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: RoPE

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self.useBidirectionalAttention = config.useBidirectionalAttention

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        let baseFreq = isSliding ? config.ropeLocalBaseFreq : config.ropeGlobalBaseFreq
        self._rope.wrappedValue = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: baseFreq
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Sliding window masking
        var finalMask = mask
        if case .array(let maskArray) = mask {
            let keySeqLen = keys.shape[2]
            if maskArray.shape.last! != keySeqLen {
                let slicedMask = maskArray[.ellipsis, (-keySeqLen)...]
                finalMask = .array(slicedMask)
            }
        }

        let maskArr: MLXArray
        if case .array(let maskArray) = finalMask {
            maskArr = maskArray
        } else {
            fatalError("oh noes")
        }
        let output = simpleSDPA(queries: queries,
                                keys: keys,
                                values: values,
                                mask: maskArr,
                                scale: scale)
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outputProj(output)
    }
}

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Attention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    let numAttentionHeads: Int
    let hiddenSize: Int
    let layerIdx: Int

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx

        self._selfAttention.wrappedValue = Attention(config, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)
        let h = Gemma.clipResidual(x, attnNorm)
        let preMLPNorm = preFeedforwardLayerNorm(h)
        let r2 = mlp(preMLPNorm)
        let postMLPNorm = postFeedforwardLayerNorm(r2)
        let out = Gemma.clipResidual(h, postMLPNorm)
        return out
    }
}

private class Gemma3Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: Gemma3TextConfiguration

    init(_ config: Gemma3TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    )
        -> MLXArray
    {
        var h: MLXArray
        h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .float32)
        h = h * scale.asType(h.dtype)
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }
        // Create attention masks
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        let j = config.slidingWindowPattern
        let globalLayerCache: [KVCache]
        if j > 0 && j <= (layerCache?.count ?? 0), let globalCache = layerCache?[j - 1] {
            globalLayerCache = [globalCache]
        } else {
            globalLayerCache = []
        }
        
        if config.useBidirectionalAttention {
            // For bidirectional attention: full attention for global layers, bidirectional sliding window for others
            var fullMaskArray = MLXArray.ones([h.dim(1), h.dim(1)], dtype: .bool)
            if case .array(let maskArray) = mask {
                fullMaskArray = fullMaskArray & maskArray
            }
            fullMask = .array(fullMaskArray)
            
            let t = h.dim(1)
            var offset = 0
            if let cache = layerCache?.compactMap({ $0 }).first {
                offset = cache.offset
            }
            var slidingWindowMaskArray = createBidirectionalSlidingWindowMask(
                n: t, offset: offset, windowSize: config.slidingWindow)
            if case .array(let maskArray) = mask {
                slidingWindowMaskArray = slidingWindowMaskArray & maskArray
            }
            slidingWindowMask = .array(slidingWindowMaskArray) 
        } else {
            // Standard causal attention
            // TODO: probably need to merge the custom mask in
            fullMask = createAttentionMask(h: h, cache: globalLayerCache)
            let allCaches = layerCache?.compactMap { $0 } ?? []
            slidingWindowMask = createAttentionMask(h: h, cache: allCaches)
        }
        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if isGlobal {
                localMask = fullMask
            } else {
                localMask = slidingWindowMask
            }
            h = layer(h, mask: localMask, cache: layerCache?[i])
        }
        return norm(h)
    }
}

public class Gemma3TextModel: Module, LLMModel {

    @ModuleInfo private var model: Gemma3Model
    @ModuleInfo(key: "lm_head") var lmHead: Module  // Can be Linear or QuantizedLinear

    public let config: Gemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = Gemma3Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray,  mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: mask, cache: cache)

        // Call the lmHead (works whether it's Linear or QuantizedLinear)
        if let linear = lmHead as? Linear {
            out = linear(out)
        } else if let quantized = lmHead as? QuantizedLinear {
            out = quantized(out)
        } else {
            fatalError("lmHead must be Linear or QuantizedLinear")
        }

        return out
    }
    
    /// Get hidden states before the language modeling head for embedding use cases
    public func getHiddenStates(_ inputs: MLXArray,  mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil, cache: [KVCache]? = nil) -> MLXArray {
        return model(inputs, mask: mask, cache: cache)
    }

    public func sanitize(
        weights: [String: MLXArray],
        quantizationConfig: QuantizationConfig? = nil
    ) -> [String: MLXArray] {
        var processedWeights = weights

        // 1. Handle VLM weight extraction first - VLM models converted using mlx_vlm.convert
        // will still have the weights under a language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // 2. Handle weight sharing (works for both regular and quantized)
        // Copy embedding weights to lm_head if lm_head weights don't exist (weight tying)
        if processedWeights["lm_head.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                let embedKey = "model.embed_tokens.\(suffix)"
                let lmHeadKey = "lm_head.\(suffix)"

                if let embedWeight = processedWeights[embedKey] {
                    processedWeights[lmHeadKey] = embedWeight
                }
            }
        }

        // 3. Apply quantization if needed
        let hasQuantizedLmHead = hasQuantizedWeights(layerPath: "lm_head", in: processedWeights)
        if hasQuantizedLmHead {
            let groupSize = quantizationConfig?.groupSize ?? 64
            let bits = quantizationConfig?.bits ?? 4

            quantize(model: self) { path, module in
                if hasQuantizedWeights(layerPath: path, in: processedWeights) {
                    return (groupSize, bits)
                }
                return nil
            }
        }

        // Remove unused precomputed rotary freqs
        return processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    /// Check if a layer has quantized weights
    private func hasQuantizedWeights(layerPath: String, in weights: [String: MLXArray]) -> Bool {
        let scalesKey = "\(layerPath).scales"
        let biasesKey = "\(layerPath).biases"
        let weightKey = "\(layerPath).weight"

        let hasScales = weights[scalesKey] != nil
        let hasBiases = weights[biasesKey] != nil
        let hasWeight = weights[weightKey]?.dtype == .uint32

        return hasScales && hasBiases && hasWeight
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                // For global layers, use standard cache but with reasonable step size for long sequences
                let cache = StandardKVCache()
                cache.step = 1024  // Larger step size for efficiency with long sequences
                caches.append(cache)
            } else {
                // For sliding window layers, use rotating cache
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }

    /// Handles prompt processing for sequences
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.shape[0]

        guard promptCount > 0 else {
            print("Warning: Preparing with empty prompt tokens.")
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }

        return .tokens(input.text)
    }
}

extension Gemma3TextModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
