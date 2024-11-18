// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXNN

class NomicEmbedding: Module {

    let typeVocabularySize: Int
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding?

    init(_ config: NomicBertConfiguration) {
        typeVocabularySize = config.typeVocabularySize
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.embedDim)
        _norm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        if config.typeVocabularySize > 0 {
            _tokenTypeEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.typeVocabularySize,
                dimensions: config.embedDim)
        }
        if config.maxPositionEmbeddings > 0 {
            _positionEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.maxPositionEmbeddings,
                dimensions: config.embedDim)
        }
    }

    func callAsFunction(
        _ inputIds: MLXArray, positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        let words = wordEmbeddings(inputIds)

        if let tokenTypeIds, let tokenTypeEmbeddings {
            words += tokenTypeEmbeddings(tokenTypeIds)
        }
        let positions =
            positionIds ?? broadcast(MLXArray.arange(inputIds.dim(1)), to: inputIds.shape)
        if let positionEmbeddings {
            words += positionEmbeddings(positions)
        }
        return norm(words)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc11") var up: Linear
    @ModuleInfo(key: "fc12") var gate: Linear
    @ModuleInfo(key: "fc2") var down: Linear

    private static func scaledHiddenFeatures(config: NomicBertConfiguration)
        -> Int
    {
        let multipleOf = 256
        let hiddenFeatures: Int = config.MLPDim
        return (hiddenFeatures + multipleOf - 1) / multipleOf * multipleOf
    }

    init(_ config: NomicBertConfiguration) {
        let hiddenFeatures = MLP.scaledHiddenFeatures(config: config)
        _up.wrappedValue = Linear(
            config.embedDim, hiddenFeatures, bias: config.mlpFc1Bias)
        _gate.wrappedValue = Linear(
            config.embedDim, hiddenFeatures, bias: config.mlpFc1Bias)
        _down.wrappedValue = Linear(
            hiddenFeatures, config.embedDim, bias: config.mlpFc2Bias)
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let activations = up(inputs) * silu(gate(inputs))
        return down(activations)
    }
}

func computeBaseFrequency(
    base: Float, dims: Int, ropeType: String,
    ropeScaling: [String: StringOrNumber]?
)
    -> Float
{
    if ropeType != "llama3" {
        return base
    }

    guard let ropeScaling = ropeScaling else {
        return base
    }

    guard case .float(let factor) = ropeScaling["factor"],
        case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"]
            ?? .float(1.0),
        case .float(let highFreqFactor) = ropeScaling["high_freq_factor"]
            ?? .float(4.0),
        case .float(let oldContextLen) = ropeScaling[
            "original_max_position_embeddings"]
            ?? .float(8192)
    else {
        return base
    }

    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    let freqs = (0 ..< dims).compactMap { index -> Float? in
        if index % 2 == 0 {
            return pow(base, Float(index) / Float(dims))
        }
        return nil
    }

    let newBaseFreqs = freqs.map { freq -> Float in
        let wavelen = 2 * .pi / freq
        let smooth = max(
            0,
            min(
                1,
                (wavelen - highFreqWavelen) / (lowFreqWavelen - highFreqWavelen)
            ))
        return freq * ((1 - smooth) * factor + smooth)
    }

    return newBaseFreqs.reduce(0, +) / Float(newBaseFreqs.count)
}

private class DynamicNTKScalingRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int?
    let traditional: Bool
    let base: Float
    var scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?

    init(
        dims: Int, maxPositionEmbeddings: Int?, traditional: Bool = false,
        base: Float = 10000, scale: Float = 1.0, ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional
        self.base = computeBaseFrequency(
            base: base, dims: dims, ropeType: ropeType, ropeScaling: ropeScaling
        )
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(1) + offset
        var base = self.base
        if let maxPositionEmbeddings, seqLen > maxPositionEmbeddings {
            let factorAdjustment =
                Float(seqLen) / Float(maxPositionEmbeddings) - 1
            let dimensionRatio = Float(dims) / Float(Float(dims) - 2)
            let adjustedScale =
                scale * pow(1 + factorAdjustment, dimensionRatio)
            base *= adjustedScale
        }
        return MLXFast.RoPE(
            x, dimensions: dims, traditional: traditional, base: base,
            scale: scale, offset: offset)
    }
}

private class Attention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "out_proj") var wo: Linear

    enum PositionalEncoding {
        case rope(RoPE)
        case dynamicNTKScalingRoPE(DynamicNTKScalingRoPE)

        func applyEncoding(_ x: MLXArray, offset: Int = 0) -> MLXArray {
            switch self {
            case .rope(let rope):
                return rope.callAsFunction(x, offset: offset)
            case .dynamicNTKScalingRoPE(let dynamicNTKScalingRoPE):
                return dynamicNTKScalingRoPE.callAsFunction(x, offset: offset)
            }
        }
    }

    let rope: PositionalEncoding
    let rotaryEmbDim: Int
    let normFactor: Float

    init(_ config: NomicBertConfiguration) {
        _wqkv.wrappedValue = Linear(
            config.embedDim, 3 * config.embedDim, bias: config.qkvProjBias)
        _wo.wrappedValue = Linear(
            config.embedDim, config.embedDim, bias: config.qkvProjBias)
        numHeads = config.numHeads
        headDim = config.embedDim / numHeads
        rotaryEmbDim = Int(Float(headDim) * config.rotaryEmbFraction)
        normFactor = sqrt(Float(headDim))

        if config.rotaryScalingFactor != nil {
            rope = .dynamicNTKScalingRoPE(
                DynamicNTKScalingRoPE(
                    dims: rotaryEmbDim,
                    maxPositionEmbeddings: config.maxPositionEmbeddings,
                    traditional: config.rotaryEmbInterleaved,
                    base: config.rotaryEmbBase,
                    scale: config.rotaryScalingFactor!))
        } else {
            rope = .rope(
                RoPE(
                    dimensions: rotaryEmbDim,
                    traditional: config.rotaryEmbInterleaved,
                    base: config.rotaryEmbBase,
                    scale: 1.0)
            )
        }
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, L) = (inputs.dim(0), inputs.dim(1))
        let queryPos = numHeads * headDim
        let qkv = split(
            wqkv(inputs), indices: [queryPos, queryPos * 2], axis: -1
        )
        var queries = qkv[0]
        var keys = qkv[1]
        var values = qkv[2]

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, numHeads, -1).transposed(
            0, 2, 1, 3)
        keys = keys.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

        if rotaryEmbDim > 0 {
            queries = rope.applyEncoding(queries)
            keys = rope.applyEncoding(keys)
        }
        var scores = queries.matmul(keys.transposed(0, 1, 3, 2)) / normFactor

        if let mask {
            scores = scores + mask
        }
        let probs = softmax(scores, axis: -1)

        let output = matmul(probs, values).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(output)
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "attn") var attention: Attention
    @ModuleInfo(key: "norm1") var postAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "norm2") var outputLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP

    init(_ config: NomicBertConfiguration) {
        _attention.wrappedValue = Attention(config)
        _mlp.wrappedValue = MLP(config)
        _outputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attentionOut = attention(inputs, mask: mask)
        let addAndNorm = postAttentionLayerNorm(attentionOut + inputs)
        let mlpOut = mlp(addAndNorm)
        return outputLayerNorm(addAndNorm + mlpOut)
    }
}

private class LMHead: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "ln") var layerNorm: LayerNorm
    @ModuleInfo(key: "decoder") var decoder: Linear

    init(_ config: NomicBertConfiguration) {
        _dense.wrappedValue = Linear(
            config.embedDim, config.embedDim, bias: config.mlpFc1Bias)
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _decoder.wrappedValue = Linear(
            config.embedDim, config.vocabularySize, bias: config.mlpFc1Bias)
    }
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        return decoder(layerNorm(silu(dense(inputs))))
    }
}

private class Encoder: Module {

    let layers: [TransformerBlock]

    init(
        _ config: NomicBertConfiguration
    ) {
        precondition(config.vocabularySize > 0)

        layers = (0 ..< config.numLayers).map {
            _ in TransformerBlock(config)
        }
    }

    func callAsFunction(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var outputs = inputs
        for (index, layer) in layers.enumerated() {
            outputs = layer(outputs, mask: attentionMask)
        }
        return outputs
    }
}

public class NomicBertModel: Module, EmbeddingModel {
    @ModuleInfo(key: "lm_head") fileprivate var lmHead: LMHead?
    @ModuleInfo(key: "embeddings") var embedder: NomicEmbedding
    let pooler: Linear?
    fileprivate let encoder: Encoder
    public var vocabularySize: Int

    public init(
        _ config: NomicBertConfiguration, pooler: Bool = true,
        lmHead: Bool = false
    ) {
        precondition(config.vocabularySize > 0)
        vocabularySize = config.vocabularySize
        encoder = Encoder(config)
        _embedder.wrappedValue = NomicEmbedding(config)

        if pooler {
            self.pooler = Linear(config.embedDim, config.embedDim)
        } else {
            self.pooler = nil
        }
        if lmHead {
            _lmHead.wrappedValue = LMHead(config)
        }
    }

    public func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    )
        -> EmbeddingModelOutput
    {
        var inp = inputs
        if inp.ndim == 1 {
            inp = inp.reshaped(1, -1)
        }
        var mask = attentionMask
        if mask != nil {
            mask = mask!.asType(embedder.wordEmbeddings.weight.dtype).expandedDimensions(axes: [
                1, 2,
            ]).log()
        }
        let outputs = encoder(
            embedder(
                inp, positionIds: positionIds, tokenTypeIds: tokenTypeIds),
            attentionMask: mask)
        if let lmHead {
            return EmbeddingModelOutput(hiddenStates: lmHead(outputs), pooledOutput: nil)
        }
        if let pooler {
            return EmbeddingModelOutput(
                hiddenStates: outputs, pooledOutput: tanh(pooler(outputs[0..., 0])))
        }
        return EmbeddingModelOutput(hiddenStates: outputs, pooledOutput: nil)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            var key = item.key.replacingOccurrences(
                of: "emb_ln", with: "embeddings.norm")
            key = key.replacingOccurrences(of: "bert.", with: "")
            key = key.replacingOccurrences(
                of: "cls.predictions.transform.dense.", with: "lm_head.dense.")
            key = key.replacingOccurrences(
                of: "cls.predictions.transform.LayerNorm.", with: "lm_head.ln.")
            key = key.replacingOccurrences(
                of: "cls.predictions.decoder", with: "lm_head.decoder")
            key = key.replacingOccurrences(of: "pooler.dense.", with: "pooler.")
            result[key] = item.value
        }
    }
}

public struct NomicBertConfiguration: Decodable, Sendable {
    var layerNormEps: Float = 1e-12
    var maxTrainedPositions: Int = 2048
    var mlpFc1Bias: Bool = false
    var mlpFc2Bias: Bool = false
    var embedDim: Int = 768
    var numHeads: Int = 12
    var MLPDim: Int = 3072
    var numLayers: Int = 12
    var qkvProjBias: Bool = false
    var rotaryEmbBase: Float = 1000
    var rotaryEmbFraction: Float = 1.0
    var rotaryEmbInterleaved: Bool = false
    var rotaryEmbScaleBase: Float?
    var rotaryScalingFactor: Float?
    var typeVocabularySize: Int = 2
    var vocabularySize: Int = 30528
    var maxPositionEmbeddings: Int = 0

    enum CodingKeys: String, CodingKey {
        case layerNormEps = "layer_norm_epsilon"
        case maxTrainedPositions = "max_trained_positions"
        case mlpFc1Bias = "mlp_fc1_bias"
        case mlpFc2Bias = "mlp_fc2_bias"
        case embedDim = "n_embd"
        case numHeads = "n_head"
        case MLPDim = "n_inner"
        case numLayers = "n_layer"
        case qkvProjBias = "qkv_proj_bias"
        case rotaryEmbBase = "rotary_emb_base"
        case rotaryEmbFraction = "rotary_emb_fraction"
        case rotaryEmbInterleaved = "rotary_emb_interleaved"
        case rotaryEmbScaleBase = "rotary_emb_scale_base"
        case rotaryScalingFactor = "rotary_scaling_factor"
        case typeVocabularySize = "type_vocab_size"
        case useCache = "use_cache"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<NomicBertConfiguration.CodingKeys> =
            try decoder.container(
                keyedBy: NomicBertConfiguration.CodingKeys.self)
        layerNormEps =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.layerNormEps.self)
            ?? 1e-12
        maxTrainedPositions =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.maxTrainedPositions
                    .self) ?? 2048
        mlpFc1Bias =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.mlpFc1Bias.self)
            ?? false
        mlpFc2Bias =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.mlpFc2Bias.self)
            ?? false
        embedDim =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.embedDim.self) ?? 768
        numHeads =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.numHeads.self) ?? 12
        MLPDim =
            try container.decodeIfPresent(
                Int.self, forKey: NomicBertConfiguration.CodingKeys.MLPDim.self)
            ?? 3072
        numLayers =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.numLayers.self) ?? 12
        qkvProjBias =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.qkvProjBias.self)
            ?? false
        rotaryEmbBase =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbBase.self)
            ?? 1000
        rotaryEmbFraction =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbFraction.self
            ) ?? 1.0
        rotaryEmbInterleaved =
            try container.decodeIfPresent(
                Bool.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbInterleaved
                    .self) ?? false
        rotaryEmbScaleBase =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryEmbScaleBase)
            ?? nil
        rotaryScalingFactor =
            try container.decodeIfPresent(
                Float.self,
                forKey: NomicBertConfiguration.CodingKeys.rotaryScalingFactor)
            ?? nil
        typeVocabularySize =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.typeVocabularySize
                    .self) ?? 2
        vocabularySize =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.vocabularySize.self)
            ?? 30528
        maxPositionEmbeddings =
            try container.decodeIfPresent(
                Int.self,
                forKey: NomicBertConfiguration.CodingKeys.maxPositionEmbeddings
                    .self) ?? 0
    }
}
