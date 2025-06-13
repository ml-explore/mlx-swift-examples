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

// MARK: - Quantization Configuration

public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

public struct Gemma3TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let ropeGlobalBaseFreq: Float
    let ropeLocalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let finalLogitSoftcapping: Float?
    let quantization: QuantizationConfig?

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
        case finalLogitSoftcapping = "final_logit_softcapping"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)

        // Default values with optional decoding
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
        quantization = try container.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

/// Clips residual connections to prevent overflow in float16 operations
private func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
    if x.dtype != .float16 {
        return x + y
    }

    // IEEE 754 half-precision maximum finite value
    let bound: Float = 65504.0  // Exactly matches mx.finfo(mx.float16).max
    let xFloat32 = x.asType(.float32)
    let yFloat32 = y.asType(.float32)
    let result = xFloat32 + yFloat32

    return clip(result, min: MLXArray(-bound), max: MLXArray(bound)).asType(.float16)
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

        // Reshape queries, keys, values
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply normalization before RoPE
        queries = queryNorm(queries)
        keys = keyNorm(keys)

        // To prevent overflow with long sequences, we upcast Q, K, and V to float32
        // and perform the entire RoPE + Attention pipeline in this higher precision.
        var queriesF32 = queries.asType(.float32)
        var keysF32 = keys.asType(.float32)
        var valuesF32 = values.asType(.float32)

        // Apply RoPE in float32
        if let cache {
            queriesF32 = rope(queriesF32, offset: cache.offset)
            keysF32 = rope(keysF32, offset: cache.offset)
            // Update the cache with the float32 keys and values
            (keysF32, valuesF32) = cache.update(keys: keysF32, values: valuesF32)
        } else {
            queriesF32 = rope(queriesF32)
            keysF32 = rope(keysF32)
        }

        // Handle sliding window masking exactly like Python
        var finalMask = mask
        if case .array(let maskArray) = mask {
            let keySeqLen = keysF32.shape[2]  // Use the float32 tensor's shape
            if maskArray.shape.last! != keySeqLen {
                let slicedMask = maskArray[.ellipsis, (-keySeqLen)...]
                finalMask = .array(slicedMask)
            }
        }

        // Manually pre-scale the float32 queries.
        let scaledQueriesF32 = queriesF32 * scale

        // Perform the entire attention calculation in float32 for stability.
        let outputF32 = MLXFast.scaledDotProductAttention(
            queries: scaledQueriesF32,
            keys: keysF32,
            values: valuesF32,
            scale: 1.0,  // Correctly set to 1.0 as we pre-scaled
            mask: finalMask
        )

        // CRITICAL: Only now do we cast the final, aggregated result of the attention
        // back to the original, potentially lower-precision data type.
        let output = outputF32.asType(x.dtype)

        let reshapedOutput = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        let finalOutput = outputProj(reshapedOutput)

        return finalOutput
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
        // Debug transformer block input for long sequences
        let seqLen = x.shape[1]
        if seqLen > 1000 {
            let xMeanTB = x.mean().item(Float.self)
            let xMaxTB = x.max().item(Float.self)
            let xMinTB = x.min().item(Float.self)
            print(
                "🔍 Layer \(layerIdx): TransformerBlock input - mean: \(xMeanTB), min: \(xMinTB), max: \(xMaxTB)"
            )

            if xMeanTB.isInfinite || xMeanTB.isNaN {
                print("🚨 Layer \(layerIdx): TransformerBlock input already has inf/NaN!")
            }
        }

        // Python: r = self.self_attn(self.input_layernorm(x), mask, cache)
        let inputNorm = inputLayerNorm(x)

        // Debug input normalization for long sequences
        if seqLen > 1000 {
            let normMean = inputNorm.mean().item(Float.self)
            let normMax = inputNorm.max().item(Float.self)
            let normMin = inputNorm.min().item(Float.self)
            print(
                "🔍 Layer \(layerIdx): After inputLayerNorm - mean: \(normMean), min: \(normMin), max: \(normMax)"
            )

            if normMean.isInfinite || normMean.isNaN {
                print("🚨 Layer \(layerIdx): inputLayerNorm produces inf/NaN!")
            }
        }

        let r = selfAttention(inputNorm, mask: mask, cache: cache)

        // Debug attention output for long sequences
        if seqLen > 1000 {
            let rMean = r.mean().item(Float.self)
            if rMean.isInfinite || rMean.isNaN {
                print("🚨 Layer \(layerIdx): Attention output has inf/NaN: \(rMean)")
            }
        }

        // Python: h = clip_residual(x, self.post_attention_layernorm(r))
        let attnNorm = postAttentionLayerNorm(r)
        let h = clipResidual(x, attnNorm)

        // Debug after attention residual for long sequences
        if seqLen > 1000 {
            let hMean = h.mean().item(Float.self)
            if hMean.isInfinite || hMean.isNaN {
                print("🚨 Layer \(layerIdx): After attention residual inf/NaN: \(hMean)")
            }
        }

        // Python: r = self.mlp(self.pre_feedforward_layernorm(h))
        let preMLPNorm = preFeedforwardLayerNorm(h)
        let r2 = mlp(preMLPNorm)

        // Debug MLP output for long sequences
        if seqLen > 1000 {
            let r2Mean = r2.mean().item(Float.self)
            if r2Mean.isInfinite || r2Mean.isNaN {
                print("🚨 Layer \(layerIdx): MLP output has inf/NaN: \(r2Mean)")
            }
        }

        // Python: out = clip_residual(h, self.post_feedforward_layernorm(r))
        let postMLPNorm = postFeedforwardLayerNorm(r2)
        let out = clipResidual(h, postMLPNorm)

        // Debug final output for long sequences
        if seqLen > 1000 {
            let outMean = out.mean().item(Float.self)
            if outMean.isInfinite || outMean.isNaN {
                print("🚨 Layer \(layerIdx): Final output has inf/NaN: \(outMean)")
            }
        }

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
        // Apply embedding scaling using float32 to prevent overflow, then convert back
        let rawEmbeddings = embedTokens(inputs)
        let embeddingScale = sqrtf(Float(config.hiddenSize))

        // This is the key:
        // 1. Upcast the embeddings to float32 to create a safe space for the multiplication.
        // 2. Perform the scaling multiplication in float32, which will not overflow.
        // 3. Cast the final, correct result back to the original data type (e.g., float16).

        let scaledF32 = rawEmbeddings.asType(.float32) * embeddingScale
        var h = scaledF32.asType(rawEmbeddings.dtype)

        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        // Create attention masks exactly like Python
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none

        if mask == nil {
            let j = config.slidingWindowPattern  // j = 6
            let globalLayerCache: [KVCache]
            if j > 0 && j <= (layerCache?.count ?? 0), let globalCache = layerCache?[j - 1] {
                globalLayerCache = [globalCache]
            } else {
                globalLayerCache = []
            }
            fullMask = createAttentionMask(h: h, cache: globalLayerCache)
            let allCaches = layerCache?.compactMap { $0 } ?? []
            slidingWindowMask = createAttentionMask(h: h, cache: allCaches)
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask = mask {
                localMask = mask
            } else if isGlobal {
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
        // Start with regular Linear - will be replaced by QuantizedLinear if quantized weights are loaded
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let optionalCache = cache?.map { $0 as KVCache? }
        let out = model(inputs, mask: nil, cache: optionalCache)

        // Call the lmHead (works whether it's Linear or QuantizedLinear)
        var finalLogits: MLXArray
        if let quantized = lmHead as? QuantizedLinear {
            finalLogits = quantized(out)
        } else if let linear = lmHead as? Linear {
            finalLogits = linear(out)
        } else {
            fatalError("lmHead must be Linear or QuantizedLinear, got: \(type(of: lmHead))")
        }

        // Apply final logit softcapping if configured
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            let scale = MLXArray(softcap)
            finalLogits = tanh(finalLogits / scale) * scale
        }

        return finalLogits
    }

    public func sanitize(weights: [String: MLXArray], quantizationConfig: QuantizationConfig? = nil)
        -> [String: MLXArray]
    {
        var processedWeights = weights

        // This function now ONLY handles weight tying. It no longer modifies the model.
        if processedWeights["lm_head.weight"] == nil {
            if let embedWeight = processedWeights["model.embed_tokens.weight"] {
                processedWeights["lm_head.weight"] = embedWeight
                print("🔗 Applied weight tying: lm_head.weight = model.embed_tokens.weight")
            }
        }

        // Remove unused precomputed rotary freqs
        return processedWeights.filter { key, _ in
            !key.contains("self_attn.rope.inv_freq")
                && !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    /// Check if a layer has quantized weights
    private func hasQuantizedWeights(layerPath: String, in weights: [String: MLXArray]) -> Bool {
        let scalesKey = "\(layerPath).scales"
        let biasesKey = "\(layerPath).biases"
        let weightKey = "\(layerPath).weight"

        return weights[scalesKey] != nil && weights[biasesKey] != nil
            && weights[weightKey]?.dtype == .uint32
    }

    // public func loraLinearLayers() -> LoRALinearLayers {
    //     model.layers.map { ($0.selfAttention, ["q_proj", "v_proj", "k_proj", "o_proj"]) } // Add k/o proj? Check common practice
    // }

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

        // Return tokens directly like Python implementation
        return .tokens(input.text)
    }
}

extension Gemma3TextModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}

extension MLXArray {
    public static func arange(_ size: Int) -> MLXArray {
        return MLXArray(Array(0 ..< size))
    }
}

/// Creates attention mask following exact mlx-lm logic
/// Exactly matches mlx-lm's create_attention_mask function
private func createAttentionMask(
    h: MLXArray,
    cache: [KVCache]?,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let T = h.shape[1]

    if T > 1 {
        var offset = 0
        var windowSize: Int? = nil
        var shouldReturnArray = returnArray

        // if cache is not None and cache[0] is not None:
        if let cache = cache, !cache.isEmpty, let firstCache = cache.first {
            // c = cache[0]
            offset = firstCache.offset

            // if hasattr(c, "max_size"):
            if let maxSize = firstCache.maxSize {
                windowSize = maxSize
                offset = min(maxSize, offset)
                // return_array = return_array or offset + T > window_size
                shouldReturnArray = shouldReturnArray || (offset + T > maxSize)
            }
        }

        // if return_array: return create_causal_mask(T, offset, window_size=window_size)
        if shouldReturnArray {
            let causalMask = createCausalMask(N: T, offset: offset, windowSize: windowSize)
            return .array(causalMask)
        } else {
            // else: return "causal"
            return .causal
        }
    } else {
        // mask = None
        return .none
    }
}

/// Creates causal mask with optional sliding window constraint
/// Exactly matches mlx-lm's create_causal_mask function
private func createCausalMask(
    N: Int,
    offset: Int = 0,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    // rinds = mx.arange(offset + N)
    let rinds = MLXArray(Int32(0) ..< Int32(offset + N))

    // linds = mx.arange(offset, offset + N) if offset else rinds
    let linds = offset > 0 ? MLXArray(Int32(offset) ..< Int32(offset + N)) : rinds

    // linds = linds[:, None], rinds = rinds[None]
    let lindsCol = linds.expandedDimensions(axis: 1)  // [N, 1]
    let rindsRow = rinds.expandedDimensions(axis: 0)  // [1, offset + N]

    // mask = linds >= rinds (causal constraint)
    var mask = lindsCol .>= rindsRow

    // if window_size is not None: mask = mask & (linds <= rinds + window_size)
    // This is the sliding window constraint - key positions must be within window_size of query position
    if let windowSize = windowSize {
        // Python: mask = mask & (linds <= rinds + window_size)
        // This means: query_pos <= key_pos + window_size
        // Combined with causal (query_pos >= key_pos), this creates a sliding window
        let windowConstraint = lindsCol .<= (rindsRow + Int32(windowSize))
        mask = MLX.logicalAnd(mask, windowConstraint)

        // Essential logging for sliding window
        if N > 1000 {
            print("🔍 Created sliding window mask: N=\(N), windowSize=\(windowSize)")
        }
    }

    // if lengths is not None: mask = mask & (rinds < lengths)
    if let lengths = lengths {
        let lengthsExpanded = lengths.expandedDimensions(axes: [1, 2, 3])  // [B, 1, 1, 1]
        let lengthConstraint = rindsRow .< lengthsExpanded
        mask = MLX.logicalAnd(mask, lengthConstraint)
    }

    // Convert boolean mask to the format expected by MLXFast
    // Python passes boolean masks directly to scaled_dot_product_attention
    // Let MLXFast handle the conversion internally, just like Python

    // Add batch and head dimensions: [1, 1, N, offset + N]
    let expandedMask = mask.expandedDimensions(axes: [0, 1])

    return expandedMask
}

/// Creates an additive sliding window mask.
///
/// Allows attention only to keys within the window size relative to the query position.
/// Mask shape: [1, 1, querySeqLen, keySeqLen]
///
/// - Parameters:
///   - querySeqLen: The sequence length of the query (L).
///   - keySeqLen: The sequence length of the key (K, potentially including offset).
///   - windowSize: The sliding window size. Must be > 0.
///   - offset: The starting position offset of the query sequence.
/// - Returns: An MLXArray suitable for adding to attention scores.
public func createAdditiveSlidingWindowMask(
    querySeqLen: Int,
    keySeqLen: Int,
    windowSize: Int,
    offset: Int = 0
) -> MLXArray {
    precondition(windowSize > 0, "windowSize must be positive for sliding window mask")

    guard querySeqLen > 0, keySeqLen > 0 else {
        // Return an empty mask if either dimension is zero
        return MLXArray.zeros([1, 1, querySeqLen, keySeqLen])
    }

    // Absolute positions of queries: [offset, offset + 1, ..., offset + L - 1]
    let queryIndices = MLXArray(Int32(offset) ..< Int32(offset + querySeqLen))  // Shape [L]

    // Absolute positions of keys: [0, 1, ..., K - 1]
    let keyIndices = MLXArray(Int32(0) ..< Int32(keySeqLen))  // Shape [K]

    // Condition: key_pos >= query_pos - windowSize
    // queryIndices shape: [L, 1]
    // keyIndices shape:   [1, K]
    // Result shape:       [L, K]
    let condition =
        keyIndices.expandedDimensions(axis: 0)
        .>= (queryIndices.expandedDimensions(axis: 1) - Int32(windowSize))

    // Create mask: 0.0 where condition is true, -inf where false
    // Use the same large negative number as causal mask for consistency
    let slidingMask = MLX.where(condition, MLXArray(0.0), MLXArray(Float(-1e9)))

    // Add dimensions for batch and head: [1, 1, L, K]
    return slidingMask.expandedDimensions(axes: [0, 1])
}
