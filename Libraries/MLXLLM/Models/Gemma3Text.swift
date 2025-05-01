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
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262208
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256  // 0.0625 ?
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
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
        mask: MLXArray? = nil,  // Keep receiving original mask (might be nil)
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

        var finalMask: MLXArray? = nil
        let currentOffset = cache?.offset ?? 0  // Get offset BEFORE cache update

        // Apply RoPE
        queries = rope(queries, offset: currentOffset)
        keys = rope(keys, offset: currentOffset)

        // Update cache and get final keys/values
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Get the TRUE key sequence length AFTER cache update
        let finalKeySeqLen = keys.dim(2)

        // **** Generate Correct Mask AFTER Cache Update ****
        // We need a mask of shape [B, 1, L, finalKeySeqLen] or broadcastable

        if L > 1 {  // Only need mask for multi-token query (prefill/chunking)
            // 1. Create the standard additive causal mask
            // Use currentOffset here as it defines the start of the query sequence
            var causalMask = createAdditiveCausalMask(n: L, offset: currentOffset)

            // 2. Slice the causal mask to match the actual key length from the cache
            // Ensure slicing doesn't go out of bounds if KVCache returns shorter keys
            let causalMaskKeyDim = causalMask.dim(3)
            let sliceDim = min(finalKeySeqLen, causalMaskKeyDim)

            // Ensure sliceDim is not negative if finalKeySeqLen or causalMaskKeyDim is 0
            let effectiveSliceDim = max(0, sliceDim)

            if effectiveSliceDim > 0 {
                causalMask = causalMask[0..., 0..., 0..., ..<effectiveSliceDim]
            } else if finalKeySeqLen > 0 {
                // Handle case where causal mask was unexpectedly empty but keys exist
                print(
                    "Warning: Causal mask empty or invalid slice (\(effectiveSliceDim)) but finalKeySeqLen=\(finalKeySeqLen). Creating zero mask."
                )
                // Create a mask of zeros (allowing all attention) and rely on sliding mask if applicable
                causalMask = MLXArray.zeros([1, 1, L, finalKeySeqLen], dtype: queries.dtype)
            } else {
                // Both are zero length, mask remains nil effectively
                causalMask = MLXArray.zeros([1, 1, L, 0], dtype: queries.dtype)  // Or handle as nil
            }

            // 3. If it's a sliding layer, create and apply the sliding window mask
            if self.isSliding {
                // Ensure slidingWindow is positive before creating mask
                let effectiveSlidingWindow = max(1, self.slidingWindow)  // Prevent non-positive window size

                let slidingMask = createAdditiveSlidingWindowMask(
                    querySeqLen: L,
                    keySeqLen: finalKeySeqLen,  // Use the final key length
                    windowSize: effectiveSlidingWindow,
                    offset: currentOffset  // Pass the query offset
                )
                // Combine masks: take the maximum (least restrictive) of the two masks
                // Since masked values are -inf, max effectively applies both constraints.
                finalMask = MLX.maximum(causalMask, slidingMask).asType(queries.dtype)
            } else {
                // Global layer: only the causal mask is needed
                finalMask = causalMask.asType(queries.dtype)
            }
        }
        // If L == 1 (single token generation), finalMask remains nil, which is correct.

        // Call attention
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: finalMask  // Use the correctly generated mask
        )
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

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize

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
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        let r2 = mlp(preFeedforwardLayerNorm(h))
        let out = h + postFeedforwardLayerNorm(r2)
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

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, cache: [KVCache?]? = nil)
        -> MLXArray
    {
        // Apply embedding with scaling
        let scale = MLXArray(sqrtf(Float(config.hiddenSize))).asType(inputs.dtype)  // Use config.hiddenSize
        var h = embedTokens(inputs) * scale

        var layerCache = cache
        if layerCache == nil {
            // During training or first pass without cache, create nil placeholders
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        for (i, layer) in layers.enumerated() {
            // Pass the original mask (or nil) directly to the layer
            h = layer(h, mask: mask, cache: layerCache?[i])
        }

        return norm(h)
    }
}

public class Gemma3TextModel: Module, LLMModel {

    @ModuleInfo private var model: Gemma3Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = Gemma3Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let optionalCache = cache?.map { $0 as KVCache? }
        let out = model(inputs, cache: optionalCache)
        var finalLogits = lmHead(out)

        // Apply final logit softcapping if configured
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            let scale = MLXArray(softcap)
            finalLogits = tanh(finalLogits / scale) * scale
        }
        return finalLogits
    }

    // TODO: Check this
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        if sanitizedWeights["lm_head.weight"] == nil {
            if let embedWeight = sanitizedWeights["model.embed_tokens.weight"] {
                sanitizedWeights["lm_head.weight"] = embedWeight
            } else {
                print("Warning: Unable to find model.embed_tokens.weight for lm_head weight tying.")
            }
        }
        // Keep filtering RoPE keys if they exist in the checkpoint (though usually not saved)
        return sanitizedWeights.filter { key, _ in
            !key.contains("self_attn.rope.inv_freq")
                && !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    // public func loraLinearLayers() -> LoRALinearLayers {
    //     model.layers.map { ($0.selfAttention, ["q_proj", "v_proj", "k_proj", "o_proj"]) } // Add k/o proj? Check common practice
    // }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                caches.append(StandardKVCache())
            } else {
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }
        return caches
    }

    /// Override the default prepare to avoid evaluating the cache, which causes crashes.
    /// Only evaluates intermediate logits during chunked prompt processing.
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.shape[0]

        guard promptCount > 0 else {
            print("Warning: Preparing with empty prompt tokens.")
            // Return empty tokens. The TokenIterator should handle this gracefully
            // by likely finishing immediately or based on how step() handles empty input.
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]  // Create an empty MLXArray of Int32
            return .tokens(.init(tokens: emptyToken))
        }

        let defaultWindowSize = 512  // Or use a value from config if available
        let effectiveWindowSize = windowSize ?? defaultWindowSize

        // Process in chunks only if the prompt is longer than 1 token AND exceeds window size
        // (No need to chunk if it fits in one window or is just 1 token)
        if promptCount > 1 && promptCount > effectiveWindowSize {
            var lastLogits: MLXArray? = nil
            // Gemma3TextModel's callAsFunction doesn't use or return separate state beyond cache
            // So we don't need to track `state` here.

            // Process all but the last token in chunks
            for i in stride(from: 0, to: promptCount - 1, by: effectiveWindowSize) {
                let start = i
                let end = min(start + effectiveWindowSize, promptCount - 1)

                guard start < end else { continue }  // Skip empty chunks if stride calculation leads to it

                // 1. Slice the token array
                let tokenSlice = promptTokens[start ..< end]
                // 2. Create LMInput.Text from the slice
                let promptChunkText = LMInput.Text(tokens: tokenSlice)

                // 3. Call the model's main forward pass using the subscript on LMInput.Text
                //    The model's callAsFunction takes MLXArray, so use [text: .newAxis]
                //    Pass nil for state as Gemma3TextModel doesn't use it.
                let result = self(promptChunkText[text: .newAxis], cache: cache, state: nil)
                lastLogits = result.logits
                // No separate state to update for Gemma3TextModel

                // IMPORTANT: Evaluate ONLY the logits, NOT the cache
                eval(result.logits)
                // No state arrays to evaluate
            }

            // If chunking happened and produced logits, return them
            if let lastLogits {
                // Return logits from processing the chunk ending at promptCount - 1
                // Pass nil for state.
                return .logits(.init(logits: lastLogits, state: nil))
            }
        }

        // If promptCount == 1 OR promptCount <= effectiveWindowSize (no chunking needed/occurred),
        // return the original tokens. The TokenIterator will call step() next using this full prompt.
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
