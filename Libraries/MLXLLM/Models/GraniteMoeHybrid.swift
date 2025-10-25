//
//  GraniteMoeHybrid.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 10/03/25.
//

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/granitemoehybrid.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum GraniteMoeHybridLayerType {
    case mamba
    case attention
}

private func createSSMMask(cache: KVCache?) -> MLXArray? {
    nil
}

private class GraniteMoeHybridRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray?) -> MLXArray {
        var states = hiddenStates
        if let gate {
            states = states * silu(gate)
        }
        return MLXFast.rmsNorm(states, weight: weight, eps: eps)
    }
}

private class GraniteMoeHybridMamba2Mixer: Module {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)

    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "norm") var norm: GraniteMoeHybridRMSNormGated

    init(_ args: GraniteMoeHybridConfiguration) {
        guard let numHeads = args.mambaHeads,
            let headDim = args.mambaHeadDim,
            let stateDim = args.mambaStateDim,
            let convKernel = args.mambaConvKernel,
            let groups = args.mambaGroups
        else {
            fatalError("GraniteMoeHybridMamba2Mixer requires Mamba parameters in the configuration")
        }

        self.numHeads = numHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = stateDim
        self.convKernelSize = convKernel
        self.intermediateSize = numHeads * headDim
        self.numGroups = groups
        self.headDim = headDim
        self.timeStepLimit = (args.timeStepMinimum, args.timeStepMaximum)
        self.convDim = intermediateSize + 2 * numGroups * ssmStateSize

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.mambaConvBias ?? false
        )

        let projectionSize = intermediateSize + convDim + numHeads
        self._inProj.wrappedValue = Linear(
            hiddenSize, projectionSize, bias: args.mambaProjBias ?? false)
        self._dtBias.wrappedValue = MLXArray.ones([numHeads])
        let headsRange = (MLXArray(0 ..< numHeads).asType(.float32) + 1)
        self._aLog.wrappedValue = MLX.log(headsRange)
        self._D.wrappedValue = MLXArray.ones([numHeads])

        self._norm.wrappedValue = GraniteMoeHybridRMSNormGated(
            dimensions: intermediateSize, eps: args.rmsNormEps)
        self._outProj.wrappedValue = Linear(
            intermediateSize, hiddenSize, bias: args.mambaProjBias ?? false)

        super.init()
    }

    private func applyConv(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let batch = input.dim(0)
        let dtype = input.dtype
        var convState = cache?[0]

        if convState == nil {
            if convKernelSize > 1 {
                convState = MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype)
            } else {
                convState = MLXArray.zeros([batch, 0, convDim], dtype: dtype)
            }
        }

        var padded = concatenated([convState!, input], axis: 1)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = padded[0..., start ..< end, 0...]
        }

        let convOutput = conv1d(padded)
        return silu(convOutput)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        var projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        var gate = splits[0]
        var convInput = splits[1]
        var dt = splits[2]

        if let mask {
            let expandedMask = expandedDimensions(mask, axis: -1)
            convInput = MLX.where(expandedMask, convInput, MLXArray.zeros(like: convInput))
        }

        let convOutput = applyConv(convInput, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )

        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        let previousState = cache?[1]
        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: previousState,
            timeStepLimit: timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = nextState
        }

        let flattenedY = y.flattened(start: 2)
        return outProj(norm(flattenedY, gate: gate))
    }
}

private class GraniteMoeHybridAttention: Module {
    let args: GraniteMoeHybridConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE?

    init(_ args: GraniteMoeHybridConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let nHeads = args.attentionHeads
        let nKvHeads = args.kvHeads
        let headDim = dim / nHeads

        self.scale = args.attentionMultiplier
        let attentionBias = args.attentionBias

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: attentionBias)
        self._wk.wrappedValue = Linear(dim, nKvHeads * headDim, bias: attentionBias)
        self._wv.wrappedValue = Linear(dim, nKvHeads * headDim, bias: attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: attentionBias)

        if args.positionEmbeddingType == "nope" {
            self.rope = nil
        } else {
            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: args.ropeTheta,
                scale: 1
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let headDim = args.hiddenSize / args.attentionHeads

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }
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

private class GraniteMoeHybridTopKGating: Module {
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "layer") var layer: Linear

    init(inputSize: Int, numExperts: Int, topK: Int) {
        self.numExperts = numExperts
        self.topK = topK
        self._layer.wrappedValue = Linear(inputSize, numExperts, bias: false)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (MLXArray, MLXArray) {
        let logits = layer(hiddenStates)
        let indices = MLX.argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let topKLogits = MLX.takeAlong(logits, indices, axis: -1)
        let gates = MLX.softmax(topKLogits, axis: -1, precise: true)
        return (indices, gates)
    }
}

private class GraniteMoeHybridMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    let router: GraniteMoeHybridTopKGating

    init(_ args: GraniteMoeHybridConfiguration) {
        guard let numExperts = args.numLocalExperts,
            let topK = args.numExpertsPerToken
        else {
            fatalError("GraniteMoeHybridMoE requires MoE parameters in the configuration")
        }

        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: numExperts,
            bias: false
        )
        self.router = GraniteMoeHybridTopKGating(
            inputSize: args.hiddenSize,
            numExperts: numExperts,
            topK: topK
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, gates) = router(x)
        let expertOutputs = switchMLP(x, indices)
        return (expertOutputs * gates[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

private class GraniteMoeHybridSharedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "output_linear") var outputLinear: Linear

    init(_ args: GraniteMoeHybridConfiguration) {
        guard let intermediate = args.sharedIntermediateSize else {
            fatalError(
                "GraniteMoeHybridSharedMLP requires shared_intermediate_size when MoE is enabled")
        }

        self._inputLinear.wrappedValue = Linear(
            args.hiddenSize, intermediate * 2, bias: false)
        self._outputLinear.wrappedValue = Linear(
            intermediate, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let splits = inputLinear(x).split(parts: 2, axis: -1)
        return outputLinear(silu(splits[0]) * splits[1])
    }
}

private class GraniteMoeHybridMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: GraniteMoeHybridConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class GraniteMoeHybridLayer: Module {
    let layerType: GraniteMoeHybridLayerType
    let residualMultiplier: Float
    let useMoE: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: GraniteMoeHybridAttention?
    @ModuleInfo(key: "mamba") var mamba: GraniteMoeHybridMamba2Mixer?
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoE: GraniteMoeHybridMoE?
    @ModuleInfo(key: "shared_mlp") var sharedMLP: GraniteMoeHybridSharedMLP?
    @ModuleInfo(key: "mlp") var mlp: GraniteMoeHybridMLP?

    init(_ args: GraniteMoeHybridConfiguration, layerType: String) {
        self.residualMultiplier = args.residualMultiplier
        self.useMoE = args.useMoE

        switch layerType {
        case "mamba":
            self.layerType = .mamba
            self._mamba.wrappedValue = GraniteMoeHybridMamba2Mixer(args)
        case "attention":
            self.layerType = .attention
            self._selfAttention.wrappedValue = GraniteMoeHybridAttention(args)
        default:
            fatalError("Unknown layer type: \(layerType)")
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if useMoE {
            self._blockSparseMoE.wrappedValue = GraniteMoeHybridMoE(args)
            self._sharedMLP.wrappedValue = GraniteMoeHybridSharedMLP(args)
        } else {
            self._mlp.wrappedValue = GraniteMoeHybridMLP(args)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        var residual = x
        var hidden = inputLayerNorm(x)

        switch layerType {
        case .mamba:
            hidden = mamba!(hidden, mask: ssmMask, cache: cache as? MambaCache)
        case .attention:
            hidden = selfAttention!(hidden, mask: attentionMask, cache: cache)
        }

        hidden = residual + hidden * residualMultiplier

        residual = hidden
        let normed = postAttentionLayerNorm(hidden)

        let mlpOutput: MLXArray
        if useMoE {
            mlpOutput = blockSparseMoE!(normed) + sharedMLP!(normed)
        } else {
            mlpOutput = mlp!(normed)
        }

        return residual + mlpOutput * residualMultiplier
    }
}

private class GraniteMoeHybridModelInner: Module {
    let args: GraniteMoeHybridConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [GraniteMoeHybridLayer]
    let norm: RMSNorm
    let embeddingMultiplier: Float
    let firstAttentionIndex: Int?
    let firstMambaIndex: Int?

    init(_ args: GraniteMoeHybridConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = args.layerTypes.map { GraniteMoeHybridLayer(args, layerType: $0) }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.embeddingMultiplier = args.embeddingMultiplier
        self.firstAttentionIndex = args.layerTypes.firstIndex(of: "attention")
        self.firstMambaIndex = args.layerTypes.firstIndex(of: "mamba")

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs) * embeddingMultiplier

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let index = firstAttentionIndex,
                let cache = cache,
                index < cache.count
            else { return .none }
            return createAttentionMask(h: hidden, cache: [cache[index]])
        }()

        let ssmMask = createSSMMask(
            cache: firstMambaIndex.flatMap { index in
                cache?[index]
            })

        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache?[i])
        }

        return norm(hidden)
    }
}

public class GraniteMoeHybridModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let logitsScaling: Float

    private let model: GraniteMoeHybridModelInner
    let configuration: GraniteMoeHybridConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: GraniteMoeHybridConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == "attention" ? args.kvHeads : 0 }
        self.logitsScaling = args.logitsScaling

        self.model = GraniteMoeHybridModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out / logitsScaling
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType in
            if layerType == "mamba" {
                return MambaCache()
            } else {
                return KVCacheSimple()
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for (key, value) in weights {
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                sanitized[key] = value.swappedAxes(1, 2)
            }
        }

        if configuration.useMoE,
            sanitized["model.layers.0.block_sparse_moe.input_linear.weight"] != nil
        {
            for layerIndex in 0 ..< configuration.hiddenLayers {
                let prefix = "model.layers.\(layerIndex).block_sparse_moe"
                guard
                    var inputWeight = sanitized.removeValue(forKey: "\(prefix).input_linear.weight")
                else { continue }

                let expertHidden = inputWeight.dim(1)
                let halfHidden = expertHidden / 2

                let gateProj = inputWeight[0..., ..<halfHidden, 0...]
                let upProj = inputWeight[0..., halfHidden..., 0...]
                sanitized["\(prefix).switch_mlp.gate_proj.weight"] = gateProj
                sanitized["\(prefix).switch_mlp.up_proj.weight"] = upProj

                if let downWeight = sanitized.removeValue(forKey: "\(prefix).output_linear.weight")
                {
                    sanitized["\(prefix).switch_mlp.down_proj.weight"] = downWeight
                }
            }
        } else if !configuration.useMoE,
            sanitized["model.layers.0.shared_mlp.input_linear.weight"] != nil
        {
            for layerIndex in 0 ..< configuration.hiddenLayers {
                let prefix = "model.layers.\(layerIndex).shared_mlp"
                guard
                    let inputWeight = sanitized.removeValue(forKey: "\(prefix).input_linear.weight")
                else { continue }

                let splits = inputWeight.split(parts: 2, axis: 0)
                sanitized["model.layers.\(layerIndex).mlp.gate_proj.weight"] = splits[0]
                sanitized["model.layers.\(layerIndex).mlp.up_proj.weight"] = splits[1]

                if let downWeight = sanitized.removeValue(forKey: "\(prefix).output_linear.weight")
                {
                    sanitized["model.layers.\(layerIndex).mlp.down_proj.weight"] = downWeight
                }
            }
        }

        return sanitized
    }

    public var loraLayers: [Module] {
        model.layers
    }
}

public struct GraniteMoeHybridConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var maxPositionEmbeddings: Int
    var attentionHeads: Int
    var kvHeads: Int
    var attentionBias: Bool
    var embeddingMultiplier: Float
    var attentionMultiplier: Float
    var logitsScaling: Float
    var residualMultiplier: Float
    var layerTypes: [String]
    var rmsNormEps: Float
    var ropeTheta: Float
    var numLocalExperts: Int?
    var numExpertsPerToken: Int?
    var sharedIntermediateSize: Int?
    var mambaHeads: Int?
    var mambaHeadDim: Int?
    var mambaProjBias: Bool?
    var mambaStateDim: Int?
    var mambaConvKernel: Int?
    var mambaGroups: Int?
    var mambaConvBias: Bool?
    var mlpBias: Bool
    var positionEmbeddingType: String
    var tieWordEmbeddings: Bool
    private let _timeStepLimit: [Float]?

    var timeStepMinimum: Float { _timeStepLimit?.first ?? 0.001 }
    var timeStepMaximum: Float { _timeStepLimit?.last ?? 100.0 }

    var useMoE: Bool { (numLocalExperts ?? 0) > 0 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case embeddingMultiplier = "embedding_multiplier"
        case attentionMultiplier = "attention_multiplier"
        case logitsScaling = "logits_scaling"
        case residualMultiplier = "residual_multiplier"
        case layerTypes = "layer_types"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case sharedIntermediateSize = "shared_intermediate_size"
        case mambaHeads = "mamba_n_heads"
        case mambaHeadDim = "mamba_d_head"
        case mambaProjBias = "mamba_proj_bias"
        case mambaStateDim = "mamba_d_state"
        case mambaConvKernel = "mamba_d_conv"
        case mambaGroups = "mamba_n_groups"
        case mambaConvBias = "mamba_conv_bias"
        case mlpBias = "mlp_bias"
        case positionEmbeddingType = "position_embedding_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _timeStepLimit = "time_step_limit"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "granitemoehybrid"
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.embeddingMultiplier = try container.decode(Float.self, forKey: .embeddingMultiplier)
        self.attentionMultiplier = try container.decode(Float.self, forKey: .attentionMultiplier)
        self.logitsScaling = try container.decode(Float.self, forKey: .logitsScaling)
        self.residualMultiplier = try container.decode(Float.self, forKey: .residualMultiplier)
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)

        self.numLocalExperts = try container.decodeIfPresent(Int.self, forKey: .numLocalExperts)
        self.numExpertsPerToken = try container.decodeIfPresent(
            Int.self, forKey: .numExpertsPerToken)
        self.sharedIntermediateSize = try container.decodeIfPresent(
            Int.self, forKey: .sharedIntermediateSize)
        self.mambaHeads = try container.decodeIfPresent(Int.self, forKey: .mambaHeads)
        self.mambaHeadDim = try container.decodeIfPresent(Int.self, forKey: .mambaHeadDim)
        self.mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias)
        self.mambaStateDim = try container.decodeIfPresent(Int.self, forKey: .mambaStateDim)
        self.mambaConvKernel = try container.decodeIfPresent(Int.self, forKey: .mambaConvKernel)
        self.mambaGroups = try container.decodeIfPresent(Int.self, forKey: .mambaGroups)
        self.mambaConvBias = try container.decodeIfPresent(Bool.self, forKey: .mambaConvBias)

        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.positionEmbeddingType =
            try container.decodeIfPresent(String.self, forKey: .positionEmbeddingType)
            ?? "rope"
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self._timeStepLimit = try container.decodeIfPresent([Float].self, forKey: ._timeStepLimit)
    }
}
