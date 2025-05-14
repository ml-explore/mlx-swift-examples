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
import ReerCodable

@Codable
public struct Gemma3TextConfiguration: Sendable {
    @CodingKey("model_type") public var modelType: String
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int = 4
    @CodingKey("head_dim") public var headDim: Int = 256
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float = 1.0e-6
    @CodingKey("vocab_size") public var vocabularySize: Int = 262144
    @CodingKey("num_key_value_heads") public var kvHeads: Int = 1
    @CodingKey("rope_global_base_freq") public var ropeGlobalBaseFreq: Float = 1_000_000.0
    @CodingKey("rope_local_base_freq") public var ropeLocalBaseFreq: Float = 10_000.0
    @CodingKey("rope_traditional") public var ropeTraditional: Bool = false
    @CodingKey("query_pre_attn_scalar") public var queryPreAttnScalar: Float = 256
    @CodingKey("sliding_window") public var slidingWindow: Int = 512
    @CodingKey("sliding_window_pattern") public var slidingWindowPattern: Int = 6
}

public struct Gemma3TextConfigurationContainer: Codable, Sendable {
    public var configuration: Gemma3TextConfiguration
    
    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: any Decoder) throws {
        // in the case of VLM models convertered using mlx_lm.convert
        // the configuration will still match the VLMs and be under text_config
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        if let configuration = try nestedContainer.decodeIfPresent(Gemma3TextConfiguration.self, forKey: .textConfig) {
            self.configuration = configuration
        } else {
            self.configuration = try Gemma3TextConfiguration(from: decoder)
        }
    }
    
    public func encode(to encoder: any Encoder) throws {
        try configuration.encode(to: encoder)
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

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: finalMask
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
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }
        // Create attention masks
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if mask == nil {
            let j = config.slidingWindowPattern
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
            if let mask {
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
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }
    
    convenience public init(_ config: Gemma3TextConfigurationContainer) {
        self.init(config.configuration)
    }

    public init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = Gemma3Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)
        out = lmHead(out)
        return out
    }

    public func sanitize(weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var processedWeights = weights

        // VLM models converted using mlx_vlm.convert will still have
        // the weights are under a language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        if processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }
        return processedWeights
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
