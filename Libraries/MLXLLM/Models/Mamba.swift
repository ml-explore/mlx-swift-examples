// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py

struct StringKey: CodingKey, ExpressibleByStringLiteral {
    var intValue: Int? = nil
    var stringValue: String
    init?(intValue: Int) { return nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init(stringLiteral: StringLiteralType) {
        self.stringValue = stringLiteral
    }
}

public struct MambaConfiguration: Codable, Sendable {
    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var stateSize: Int
    var numHiddenLayers: Int
    var convKernel: Int
    var useBias: Bool
    var useConvBias: Bool
    var timeStepRank: Int
    var tieWordEmbeddings: Bool
    var useBcdtRms: Bool
    var mixerRmsEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case stateSize = "state_size"
        case numHiddenLayers = "num_hidden_layers"
        case convKernel = "conv_kernel"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case timeStepRank = "time_step_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBcdtRms = "use_bcdt_rms"
        case mixerRmsEps = "mixer_rms_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = try decoder.container(keyedBy: StringKey.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize =
            try container
            .decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? fallback
            .decode(Int.self, forKey: "d_model")
        intermediateSize =
            try container
            .decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? fallback
            .decode(Int.self, forKey: "d_inner")
        stateSize =
            try container
            .decodeIfPresent(Int.self, forKey: .stateSize)
            ?? fallback
            .decode(Int.self, forKey: "d_state")
        numHiddenLayers =
            try container
            .decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? fallback
            .decodeIfPresent(Int.self, forKey: "n_layer")
            ?? fallback
            .decode(Int.self, forKey: "n_layers")
        convKernel =
            try container
            .decodeIfPresent(Int.self, forKey: .convKernel)
            ?? fallback
            .decode(Int.self, forKey: "d_conv")
        useBias =
            try container
            .decodeIfPresent(Bool.self, forKey: .useBias)
            ?? fallback
            .decode(Bool.self, forKey: "bias")
        useConvBias =
            try container
            .decodeIfPresent(Bool.self, forKey: .useConvBias)
            ?? fallback
            .decode(Bool.self, forKey: "conv_bias")

        if let timeStepRankAuto = try? container.decode(String.self, forKey: .timeStepRank),
            timeStepRankAuto == "auto"
        {
            timeStepRank = (hiddenSize + 15) / 16
        } else {
            timeStepRank = try container.decode(Int.self, forKey: .timeStepRank)
        }

        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        useBcdtRms = try container.decodeIfPresent(Bool.self, forKey: .useBcdtRms) ?? false
        mixerRmsEps = try container.decodeIfPresent(Float.self, forKey: .mixerRmsEps) ?? 1e-6

        if modelType == "falcon_mamba" {
            useBcdtRms = true
        }
    }
}

private class MambaBlock: Module {

    let args: MambaConfiguration

    var _mixerNorm: ((MLXArray) -> MLXArray)? = nil

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray

    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(_ args: MambaConfiguration) {
        self.args = args
        if args.useBcdtRms {
            self._mixerNorm = {
                MLXFast.rmsNorm(
                    $0,
                    weight: MLX.ones([$0.dim(-1)], dtype: $0.dtype),
                    eps: args.mixerRmsEps)
            }
        }

        self._inProj.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize * 2, bias: args.useBias)

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: args.intermediateSize,
            outputChannels: args.intermediateSize,
            kernelSize: args.convKernel,
            padding: 0,
            groups: args.intermediateSize,
            bias: args.useConvBias
        )

        self._xProj.wrappedValue = Linear(
            args.intermediateSize,
            args.timeStepRank + 2 * args.stateSize,
            bias: false
        )

        self._dtProj.wrappedValue = Linear(
            args.timeStepRank, args.intermediateSize, bias: true)

        let A = repeated(
            MLXArray(1 ..< args.stateSize + 1, [1, args.stateSize]),
            count: args.intermediateSize,
            axis: 0
        )

        self._aLog.wrappedValue = log(A)
        self._d.wrappedValue = ones([args.intermediateSize])

        self._outProj.wrappedValue = Linear(
            args.intermediateSize, args.hiddenSize, bias: args.useBias)
    }

    func ssmStep(_ x: MLXArray, _ A: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let deltaBC = self.xProj(x)
        var deltaBCParts = split(
            deltaBC,
            indices: [self.args.timeStepRank, self.args.timeStepRank + self.args.stateSize],
            axis: -1
        ).map {
            if self.args.useBcdtRms, let mixerNorm = self._mixerNorm {
                return mixerNorm($0)
            } else {
                return $0
            }
        }
        if self.args.useBcdtRms, let mixerNorm = self._mixerNorm {
            deltaBCParts = deltaBCParts.map { mixerNorm($0) }
        }
        var delta = deltaBCParts[0]
        let B = deltaBCParts[1]
        let C = deltaBCParts[2]

        delta = softplus(self.dtProj(delta))
        var newState = expandedDimensions(delta * x, axis: -1) * expandedDimensions(B, axis: 1)
        if let state {
            newState += state * exp(expandedDimensions(delta, axis: -1) * A)
        }
        var y = newState.matmul(expandedDimensions(C, axis: -1)).squeezed(axis: 2)
        y = y + self._d.wrappedValue * x
        return (y, newState)
    }

    func processSequence(_ x: MLXArray, convCache: MLXArray?, stateCache: MLXArray?)
        -> (MLXArray, (MLXArray, MLXArray?))
    {
        let T = x.dim(1)
        let xz = self.inProj(x)
        var (x, z) = xz.split(axis: -1)
        let K = self.args.convKernel
        var xFull: MLXArray
        if let convCache {
            xFull = concatenated([convCache, x], axis: 1)
        } else {
            xFull = padded(
                x,
                widths: [
                    .init((0, 0)),
                    .init((K - 1, 0)),
                    .init((0, 0)),
                ])
        }
        let convOut = conv1d(xFull)
        // TODO there is a failure in the next line, maybe need .newAxis or something
        // there are only 3 slices in the python code, not 4
        // I need to figure out how to transalte -(K-1)... to swift
        // the following compiles, but not sure if it is correct
        let newConvCache = xFull[0..., (1 - K)..., 0...]
        x = silu(convOut)
        let A = -exp(self.aLog)
        var currentState = stateCache
        var y: [MLXArray] = []
        var yT: MLXArray
        for t in 0 ..< T {
            (yT, currentState) = self.ssmStep(x[0..., t], A, state: currentState)
            y.append(yT)
        }
        z = self.outProj(silu(z) * stacked(y, axis: 1))
        return (z, (newConvCache, currentState))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: MambaCache? = nil) -> MLXArray {
        let (output, (newConvCache, newStateCache)) = self.processSequence(
            inputs, convCache: cache?[0], stateCache: cache?[1]
        )
        if cache != nil {
            cache![0] = newConvCache
            cache![1] = newStateCache
        }
        return output
    }

}

private class ResidualBlock: Module {
    @ModuleInfo var mixer: MambaBlock
    @ModuleInfo var norm: RMSNorm
    public init(_ args: MambaConfiguration) {
        self._mixer.wrappedValue = MambaBlock(args)
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: MambaCache? = nil) -> MLXArray {
        return mixer(norm(inputs), cache: cache) + inputs
    }
}

// maps to mamba.Mamba
private class MambaModelInner: Module {
    @ModuleInfo var embeddings: Embedding
    @ModuleInfo var layers: [ResidualBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    public init(_ args: MambaConfiguration) {
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self._layers.wrappedValue = (0 ..< args.numHiddenLayers).map { _ in
            ResidualBlock(args)
        }
        self._normF.wrappedValue = RMSNorm(dimensions: args.hiddenSize)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var x = embeddings(inputs)
        for (i, layer) in layers.enumerated() {
            x = layer(x, cache: (cache?[i] as? MambaCache))
        }
        return normF(x)
    }
}

// maps to mamba.Model
public class MambaModel: Module, LLMModel {
    let args: MambaConfiguration
    let modelType: String
    @ModuleInfo private var backbone: MambaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear? = nil

    public init(_ args: MambaConfiguration) {
        self.args = args
        self.modelType = args.modelType
        self._backbone.wrappedValue = MambaModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let x = self.backbone(inputs, cache: cache)
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(x)
        } else {
            logits = self.backbone.embeddings.asLinear(x)
        }
        return logits
    }

    public func newCache(parameters: MLXLMCommon.GenerateParameters?)
        -> [any MLXLMCommon.KVCache]
    {
        return (0 ..< args.numHiddenLayers).map { _ in MambaCache() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights
        for (key, value) in weights {
            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                processedWeights[key] = value.movedAxis(source: 2, destination: 1)
            }
        }
        return processedWeights
    }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        // TODO ???
        return []
    }

}
