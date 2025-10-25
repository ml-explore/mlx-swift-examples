//
//  FalconH1.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/6/18.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/falcon_h1.py

// MARK: - Configuration

public struct FalconH1Configuration: Codable, Sendable {
    var attentionBias: Bool
    var attentionDropout: Float
    var attentionInMultiplier: Float
    var attentionOutMultiplier: Float
    var bosTokenId: Int
    var embeddingMultiplier: Float
    var eosTokenId: Int
    var headDim: Int
    var hiddenAct: String
    var hiddenSize: Int
    var initializerRange: Float
    var intermediateSize: Int?
    var keyMultiplier: Float
    var lmHeadMultiplier: Float
    var mambaChunkSize: Int
    var mambaConvBias: Bool
    var mambaDConv: Int
    var mambaDHead: Int
    var mambaDSSM: Int
    var mambaDState: Int
    var mambaExpand: Int
    var mambaNGroups: Int
    var mambaNHeads: Int
    var mambaNormBeforeGate: Bool
    var mambaProjBias: Bool
    var mambaRMSNorm: Bool
    var mambaUseMLP: Bool
    var maxPositionEmbeddings: Int
    var mlpBias: Bool
    var mlpExpansionFactor: Int
    var mlpMultipliers: [Float]
    var modelType: String
    var numAttentionHeads: Int
    var numHiddenLayers: Int
    var numKeyValueHeads: Int
    var numLogitsToKeep: Int
    var padTokenId: Int
    var projectorsBias: Bool
    var rmsNormEps: Float
    var ropeTraditional: Bool
    var ropeScaling: Float?
    var ropeTheta: Float
    var ssmInMultiplier: Float
    var ssmMultipliers: [Float]
    var ssmOutMultiplier: Float
    var tieWordEmbeddings: Bool
    var torchDtype: String
    var vocabSize: Int

    enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case attentionInMultiplier = "attention_in_multiplier"
        case attentionOutMultiplier = "attention_out_multiplier"
        case bosTokenId = "bos_token_id"
        case embeddingMultiplier = "embedding_multiplier"
        case eosTokenId = "eos_token_id"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case initializerRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case keyMultiplier = "key_multiplier"
        case lmHeadMultiplier = "lm_head_multiplier"
        case mambaChunkSize = "mamba_chunk_size"
        case mambaConvBias = "mamba_conv_bias"
        case mambaDConv = "mamba_d_conv"
        case mambaDHead = "mamba_d_head"
        case mambaDSSM = "mamba_d_ssm"
        case mambaDState = "mamba_d_state"
        case mambaExpand = "mamba_expand"
        case mambaNGroups = "mamba_n_groups"
        case mambaNHeads = "mamba_n_heads"
        case mambaNormBeforeGate = "mamba_norm_before_gate"
        case mambaProjBias = "mamba_proj_bias"
        case mambaRMSNorm = "mamba_rms_norm"
        case mambaUseMLP = "mamba_use_mlp"
        case maxPositionEmbeddings = "max_position_embeddings"
        case mlpBias = "mlp_bias"
        case mlpExpansionFactor = "mlp_expansion_factor"
        case mlpMultipliers = "mlp_multipliers"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numLogitsToKeep = "num_logits_to_keep"
        case padTokenId = "pad_token_id"
        case projectorsBias = "projectors_bias"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case ssmInMultiplier = "ssm_in_multiplier"
        case ssmMultipliers = "ssm_multipliers"
        case ssmOutMultiplier = "ssm_out_multiplier"
        case tieWordEmbeddings = "tie_word_embeddings"
        case torchDtype = "torch_dtype"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.attentionInMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .attentionInMultiplier) ?? 1.0
        self.attentionOutMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .attentionOutMultiplier) ?? 1.0
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.embeddingMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1.0
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.initializerRange =
            try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? nil
        self.keyMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .keyMultiplier) ?? 1.0
        self.lmHeadMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .lmHeadMultiplier) ?? 1.0
        self.mambaChunkSize =
            try container.decodeIfPresent(Int.self, forKey: .mambaChunkSize) ?? 256
        self.mambaConvBias =
            try container.decodeIfPresent(Bool.self, forKey: .mambaConvBias) ?? true
        self.mambaDConv = try container.decodeIfPresent(Int.self, forKey: .mambaDConv) ?? 4
        self.mambaDHead = try container.decodeIfPresent(Int.self, forKey: .mambaDHead) ?? 64
        self.mambaDSSM = try container.decodeIfPresent(Int.self, forKey: .mambaDSSM) ?? 1536
        self.mambaDState = try container.decodeIfPresent(Int.self, forKey: .mambaDState) ?? 256
        self.mambaExpand = try container.decodeIfPresent(Int.self, forKey: .mambaExpand) ?? 2
        self.mambaNGroups = try container.decodeIfPresent(Int.self, forKey: .mambaNGroups) ?? 1
        self.mambaNHeads = try container.decodeIfPresent(Int.self, forKey: .mambaNHeads) ?? 128
        self.mambaNormBeforeGate =
            try container.decodeIfPresent(Bool.self, forKey: .mambaNormBeforeGate) ?? true
        self.mambaProjBias =
            try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        self.mambaRMSNorm = try container.decodeIfPresent(Bool.self, forKey: .mambaRMSNorm) ?? false
        self.mambaUseMLP = try container.decodeIfPresent(Bool.self, forKey: .mambaUseMLP) ?? true
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.mlpExpansionFactor =
            try container.decodeIfPresent(Int.self, forKey: .mlpExpansionFactor) ?? 8
        self.mlpMultipliers =
            try container.decodeIfPresent([Float].self, forKey: .mlpMultipliers) ?? [1.0, 1.0]
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_h1"
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.numLogitsToKeep =
            try container.decodeIfPresent(Int.self, forKey: .numLogitsToKeep) ?? 1
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        self.projectorsBias =
            try container.decodeIfPresent(Bool.self, forKey: .projectorsBias) ?? false
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(Float?.self, forKey: .ropeScaling) ?? nil
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
        self.ssmInMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .ssmInMultiplier) ?? 1.0
        self.ssmMultipliers =
            try container.decodeIfPresent([Float].self, forKey: .ssmMultipliers) ?? [
                1.0, 1.0, 1.0, 1.0, 1.0,
            ]
        self.ssmOutMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .ssmOutMultiplier) ?? 1.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.torchDtype =
            try container.decodeIfPresent(String.self, forKey: .torchDtype) ?? "bfloat16"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128000
    }
}

// MARK: - RMSNormGated

private class RMSNormGated: Module {
    let weight: MLXArray
    let varianceEpsilon: Float
    let nGroups: Int
    let normBeforeGate: Bool

    init(hiddenSize: Int, eps: Float = 1e-6, nGroups: Int = 1, normBeforeGate: Bool = true) {
        self.weight = MLXArray.ones([hiddenSize])
        self.varianceEpsilon = eps
        self.nGroups = nGroups
        self.normBeforeGate = normBeforeGate
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        var hiddenStates = hiddenStates

        if !normBeforeGate, let gate {
            hiddenStates = hiddenStates * silu(gate)
        }

        hiddenStates = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: varianceEpsilon)

        if normBeforeGate, let gate {
            hiddenStates = hiddenStates * silu(gate)
        }

        return hiddenStates
    }
}

private func computeMupVector(_ args: FalconH1Configuration) -> MLXArray {
    let intermediateSize = args.mambaDSSM
    let groupsTimeStateSize = args.mambaNGroups * args.mambaDState
    let numHeads = args.mambaNHeads

    let sizes = [
        intermediateSize,
        intermediateSize,
        groupsTimeStateSize,
        groupsTimeStateSize,
        numHeads,
    ]

    let segments = zip(sizes, args.ssmMultipliers).map { size, multiplier in
        MLX.broadcast(MLXArray(multiplier), to: [size])
    }

    return concatenated(segments)
}

// MARK: - Attention

private class Attention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ args: FalconH1Configuration) {
        self.hiddenSize = args.hiddenSize
        self.numHeads = args.numAttentionHeads
        self.numKVHeads = args.numKeyValueHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: args.attentionBias)

        let ropeScale: Float =
            if let ropeScaling = args.ropeScaling {
                1 / ropeScaling
            } else {
                1
            }

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale
        )
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }
}

// MARK: - Mixer

private class Mixer: Module {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let useConvBias: Bool
    let useBias: Bool
    let layerNormEpsilon: Float
    let groupsTimeStateSize: Int
    let nGroups: Int
    let headDim: Int
    let chunkSize: Int
    let timeStepLimit: (Float, Float)
    let timeStepMin: Float
    let timeStepMax: Float
    let convDim: Int
    let mambaRMSNorm: Bool
    var norm: RMSNormGated? = nil
    let ssmInMultiplier: Float
    let conv1d: Conv1d

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: FalconH1Configuration) {
        self.numHeads = args.mambaNHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = args.mambaDState
        self.convKernelSize = args.mambaDConv
        self.intermediateSize = args.mambaDSSM
        self.useConvBias = args.mambaConvBias
        self.useBias = args.mambaProjBias
        self.layerNormEpsilon = args.rmsNormEps
        self.groupsTimeStateSize = args.mambaNGroups * args.mambaDState
        self.nGroups = args.mambaNGroups
        self.headDim = args.mambaDHead
        self.chunkSize = args.mambaChunkSize
        self.timeStepLimit = (0.0, Float.infinity)
        self.timeStepMin = 0.001
        self.timeStepMax = 0.1

        self.convDim = intermediateSize + 2 * nGroups * ssmStateSize

        self.conv1d = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        _inProj.wrappedValue = Linear(
            hiddenSize,
            projectionSize,
            bias: args.mambaProjBias
        )

        _dtBias.wrappedValue = MLXArray.ones([numHeads])

        let A = MLXArray(Array(1 ..< numHeads + 1))

        _aLog.wrappedValue = log(A)

        self.mambaRMSNorm = args.mambaRMSNorm
        if mambaRMSNorm {
            self.norm = RMSNormGated(
                hiddenSize: intermediateSize,
                eps: layerNormEpsilon,
                nGroups: nGroups,
                normBeforeGate: args.mambaNormBeforeGate
            )
        }

        _d.wrappedValue = MLXArray.ones([numHeads])

        _outProj.wrappedValue = Linear(
            intermediateSize,
            hiddenSize,
            bias: args.projectorsBias
        )

        self.ssmInMultiplier = args.ssmInMultiplier
    }

    private func _applyConv(_ convInput: MLXArray, cache: MambaCache?) -> MLXArray {
        let convState: MLXArray
        if cache == nil || cache?[0] == nil {
            convState = MLXArray.zeros(
                [convInput.dim(0), convKernelSize - 1, convDim],
                dtype: convInput.dtype
            )
        } else {
            convState = cache![0]!
        }

        let paddedInput = concatenated([convState, convInput], axis: 1)

        if let cache = cache {
            cache[0] = paddedInput[0..., (-(convKernelSize - 1))...]
        }

        let convOutput = conv1d(paddedInput)
        return silu(convOutput)
    }

    private func _ssm(
        hiddenStates: MLXArray,
        B: MLXArray,
        C: MLXArray,
        dt: MLXArray,
        state: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (batchSize, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        let hiddenStates = hiddenStates.reshaped(batchSize, seqLen, numHeads, headDim)
        let B = B.reshaped(batchSize, seqLen, nGroups, ssmStateSize)
        let C = C.reshaped(batchSize, seqLen, nGroups, ssmStateSize)

        let (y, newState) = ssmUpdate(
            hiddenStates: hiddenStates,
            ALog: aLog,
            B: B,
            C: C,
            D: d,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit,
            mask: mask
        )

        return (y.reshaped(batchSize, seqLen, intermediateSize), newState)
    }

    func callAsFunction(
        _ inputStates: MLXArray, cache: MambaCache? = nil, mask: MLXArray? = nil
    ) -> MLXArray {
        let projectedStates = inProj(inputStates)

        let splits = MLX.split(
            projectedStates,
            indices: [intermediateSize, intermediateSize + convDim],
            axis: -1
        )
        let gate = splits[0]
        var convInput = splits[1]
        let dt = splits[2]

        if let mask = mask {
            convInput = which(mask[.ellipsis, .newAxis], convInput, 0)
        }
        let convOutput = _applyConv(convInput, cache: cache)

        let convSplits = MLX.split(
            convOutput,
            indices: [
                intermediateSize,
                intermediateSize + nGroups * ssmStateSize,
            ],
            axis: -1
        )
        let hiddenStatesSSM = convSplits[0]
        let B = convSplits[1]
        let C = convSplits[2]

        var state = cache?[1]
        var y: MLXArray
        (y, state) = _ssm(
            hiddenStates: hiddenStatesSSM,
            B: B,
            C: C,
            dt: dt,
            state: state,
            mask: mask
        )
        if let cache = cache {
            cache[1] = state
        }

        if let norm = norm {
            y = norm(y, gate: gate)
        } else {
            y = y * silu(gate)
        }

        return outProj(y)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let gateMultiplier: Float
    let downMultiplier: Float

    init(_ args: FalconH1Configuration) {
        let hiddenSize = args.hiddenSize
        let intermediateSize = args.intermediateSize ?? 4 * hiddenSize

        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: args.mlpBias)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: args.mlpBias)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: args.mlpBias)

        self.gateMultiplier = args.mlpMultipliers[0]
        self.downMultiplier = args.mlpMultipliers[1]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = upProj(x) * silu(gateProj(x))
        return downProj(y)
    }
}

// MARK: - DecoderLayer

private class DecoderLayer: Module {
    @ModuleInfo(key: "feed_forward") var feedForward: MLP
    @ModuleInfo(key: "mamba") var mamba: Mixer
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_ff_layernorm") var preFfLayerNorm: RMSNorm

    let channelsAttn: Int

    init(_ args: FalconH1Configuration) {
        let headDim = args.headDim
        self.channelsAttn = args.numAttentionHeads * headDim + 2 * args.numKeyValueHeads * headDim

        _feedForward.wrappedValue = MLP(args)
        _mamba.wrappedValue = Mixer(args)
        _attention.wrappedValue = Attention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps
        )
        _preFfLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps
        )
    }

    func callAsFunction(
        _ h: MLXArray,
        cache: CacheList?,
        attnMask: MLXArray?,
        mambaMask: MLXArray?
    ) -> MLXArray {
        var residual = h
        var h = inputLayerNorm(h)

        let mambaH = mamba(h, cache: cache?[0] as? MambaCache, mask: mambaMask)

        let attnH = attention(
            h,
            mask: attnMask,
            cache: cache?[1]
        )

        h = residual + mambaH + attnH

        residual = h
        h = preFfLayerNorm(h)
        h = feedForward(h)
        return residual + h
    }
}

// MARK: - Helper Functions

private func createSSMMask(h: MLXArray, cache: ArraysCache?) -> MLXArray? {
    if let cache = cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

private func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let N = h.dim(1)
    // If cache exists and can make masks, use it
    // Otherwise for single token, no mask needed
    // For multi-token, SDPA will handle causal mask internally when nil
    if N == 1 {
        return nil
    }
    return nil  // Will be handled by SDPA internally when nil
}

// MARK: - Model

private class ModelInner: Module {
    let args: FalconH1Configuration
    let vocabSize: Int
    let hiddenSize: Int

    let _mupVector: MLXArray
    let layers: [DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm

    init(_ args: FalconH1Configuration) {
        self.args = args
        self.vocabSize = args.vocabSize
        self.hiddenSize = args.hiddenSize

        _embedTokens.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)

        self._mupVector = computeMupVector(args)
        self.layers = (0 ..< args.numHiddenLayers).map { _ in
            DecoderLayer(args)
        }

        _finalLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, cache: [CacheList]? = nil)
        -> MLXArray
    {
        var h = embedTokens(inputs)

        let cache: [CacheList?] = cache ?? Array(repeating: nil, count: layers.count)

        let mambaMask = createSSMMask(h: h, cache: cache[0]?[0] as? MambaCache)
        let attnMask: MLXArray? = createAttentionMask(
            h: h, cache: cache[0]?[1] != nil ? [cache[0]![1]] : nil)

        for (layer, c) in zip(layers, cache) {
            h = layer(
                h,
                cache: c,
                attnMask: attnMask,
                mambaMask: mambaMask
            )
        }

        return finalLayerNorm(h)
    }
}

public class FalconH1Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: ModelInner
    let configuration: FalconH1Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: FalconH1Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabSize
        self.kvHeads = (0 ..< args.numKeyValueHeads).map { _ in args.numHiddenLayers }
        self.model = ModelInner(args)

        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache as? [CacheList])
        return lmHead(out)
    }

    public func makeCache() -> [CacheList] {
        return (0 ..< configuration.numHiddenLayers).map { _ in
            CacheList(MambaCache(), KVCacheSimple())
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let c1d = weights["model.layers.0.mamba.conv1d.weight"]!
        if c1d.dim(-1) <= c1d.dim(1) {
            return weights
        }

        var sanitizedWeights = [String: MLXArray]()
        let args = configuration

        for (name, var param) in weights {
            if name.hasSuffix("embed_tokens.weight") {
                param = param * args.embeddingMultiplier
            } else if name.hasSuffix("lm_head.weight") {
                param = param * args.lmHeadMultiplier
            } else if name.hasSuffix("q_proj.weight") || name.hasSuffix("k_proj.weight") {
                param = param * args.attentionInMultiplier
            } else if name.hasSuffix("key_proj.weight") {
                param = param * args.attentionInMultiplier * args.keyMultiplier
            } else if name.hasSuffix("o_proj.weight") {
                param = param * args.attentionOutMultiplier
            } else if name.hasSuffix("out_proj.weight") {
                param = param * args.ssmOutMultiplier
            } else if name.hasSuffix("gate_proj.weight") {
                param = param * args.mlpMultipliers[0]
            } else if name.hasSuffix("down_proj.weight") {
                param = param * args.mlpMultipliers[1]
            } else if name.hasSuffix("in_proj.weight") {
                param =
                    param
                    * (args.ssmInMultiplier * model._mupVector.asType(param.dtype)[0..., .newAxis])
            } else if name.contains("conv1d.weight") {
                param = param.transposed(0, 2, 1)
            }

            sanitizedWeights[name] = param
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        model.layers.map { _ in CacheList(MambaCache(), KVCacheSimple()) }
    }
}

// MARK: - LoRA

extension FalconH1Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
