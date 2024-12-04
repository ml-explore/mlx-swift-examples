// Copyright © 2024 Apple Inc.

import MLX
import MLXFast
import MLXNN

extension MLXArray {
    public static func arange(_ size: Int) -> MLXArray {
        return MLXArray(Array(0 ..< size))
    }
}

private class BertEmbedding: Module {

    let typeVocabularySize: Int
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding

    init(_ config: BertConfiguration) {
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
        _positionEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.embedDim)

    }

    func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        let posIds = positionIds ?? broadcast(MLXArray.arange(inputIds.dim(1)), to: inputIds.shape)
        let words = wordEmbeddings(inputIds) + positionEmbeddings(posIds)
        if let tokenTypeIds, let tokenTypeEmbeddings {
            words += tokenTypeEmbeddings(tokenTypeIds)
        }
        return norm(words)
    }
}

private class TransformerBlock: Module {
    let attention: MultiHeadAttention
    @ModuleInfo(key: "ln1") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "ln2") var postLayerNorm: LayerNorm
    @ModuleInfo(key: "linear1") var up: Linear
    @ModuleInfo(key: "linear2") var down: Linear

    init(_ config: BertConfiguration) {
        attention = MultiHeadAttention(
            dimensions: config.embedDim, numHeads: config.numHeads, bias: true)
        _preLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _up.wrappedValue = Linear(config.embedDim, config.interDim)
        _down.wrappedValue = Linear(config.interDim, config.embedDim)
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attentionOut = attention(inputs, keys: inputs, values: inputs, mask: mask)
        let preNorm = preLayerNorm(inputs + attentionOut)

        let mlpOut = down(gelu(up(preNorm)))
        return postLayerNorm(mlpOut + preNorm)
    }
}

private class Encoder: Module {
    let layers: [TransformerBlock]
    init(_ config: BertConfiguration) {
        precondition(config.vocabularySize > 0)
        layers = (0 ..< config.numLayers).map { _ in TransformerBlock(config) }
    }
    func callAsFunction(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var outputs = inputs
        for layer in layers {
            outputs = layer(outputs, mask: attentionMask)
        }
        return outputs
    }
}

private class LMHead: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "ln") var layerNorm: LayerNorm
    @ModuleInfo(key: "decoder") var decoder: Linear

    init(_ config: BertConfiguration) {
        _dense.wrappedValue = Linear(
            config.embedDim, config.embedDim, bias: true)
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.embedDim, eps: config.layerNormEps)
        _decoder.wrappedValue = Linear(
            config.embedDim, config.vocabularySize, bias: true)
    }
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        return decoder(layerNorm(silu(dense(inputs))))
    }
}

public class BertModel: Module, EmbeddingModel {
    @ModuleInfo(key: "lm_head") fileprivate var lmHead: LMHead?
    @ModuleInfo(key: "embeddings") fileprivate var embedder: BertEmbedding
    let pooler: Linear?
    fileprivate let encoder: Encoder
    public var vocabularySize: Int

    public init(
        _ config: BertConfiguration, lmHead: Bool = false
    ) {
        precondition(config.vocabularySize > 0)
        vocabularySize = config.vocabularySize
        encoder = Encoder(config)
        _embedder.wrappedValue = BertEmbedding(config)

        if lmHead {
            _lmHead.wrappedValue = LMHead(config)
            self.pooler = nil
        } else {
            pooler = Linear(config.embedDim, config.embedDim)
            _lmHead.wrappedValue = nil
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
            embedder(inp, positionIds: positionIds, tokenTypeIds: tokenTypeIds),
            attentionMask: mask)
        if let lmHead {
            return EmbeddingModelOutput(hiddenStates: lmHead(outputs), pooledOutput: nil)
        } else {
            return EmbeddingModelOutput(
                hiddenStates: outputs, pooledOutput: tanh(pooler!(outputs[0..., 0])))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            var key = item.key.replacingOccurrences(of: ".layer.", with: ".layers.")
            key = key.replacingOccurrences(of: ".self.key.", with: ".key_proj.")
            key = key.replacingOccurrences(of: ".self.query.", with: ".query_proj.")
            key = key.replacingOccurrences(of: ".self.value.", with: ".value_proj.")
            key = key.replacingOccurrences(
                of: ".attention.output.dense.", with: ".attention.out_proj.")
            key = key.replacingOccurrences(of: ".attention.output.LayerNorm.", with: ".ln1.")
            key = key.replacingOccurrences(of: ".output.LayerNorm.", with: ".ln2.")
            key = key.replacingOccurrences(of: ".intermediate.dense.", with: ".linear1.")
            key = key.replacingOccurrences(of: ".output.dense.", with: ".linear2.")
            key = key.replacingOccurrences(of: ".LayerNorm.", with: ".norm.")
            key = key.replacingOccurrences(of: "pooler.dense.", with: "pooler.")
            key = key.replacingOccurrences(
                of:
                    "cls.predictions.transform.dense.",
                with: "lm_head.dense.")
            key = key.replacingOccurrences(
                of:
                    "cls.predictions.transform.LayerNorm.",
                with: "lm_head.ln.")
            key = key.replacingOccurrences(
                of:
                    "cls.predictions.decoder",
                with: "lm_head.decoder")
            key = key.replacingOccurrences(
                of: "cls.predictions.transform.norm.weight",
                with: "lm_head.ln.weight")
            key = key.replacingOccurrences(
                of: "cls.predictions.transform.norm.bias",
                with: "lm_head.ln.bias")
            key = key.replacingOccurrences(of: "cls.predictions.bias", with: "lm_head.decoder.bias")
            key = key.replacingOccurrences(of: "bert.", with: "")
            result[key] = item.value
        }.filter { key, _ in key != "embeddings.position_ids" }
    }
}

public class DistilBertModel: BertModel {
    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            var key = item.key.replacingOccurrences(of: ".layer.", with: ".layers.")
            key = key.replacingOccurrences(of: "transformer.", with: "encoder.")
            key = key.replacingOccurrences(of: "embeddings.LayerNorm", with: "embeddings.norm")
            key = key.replacingOccurrences(of: ".attention.q_lin.", with: ".attention.query_proj.")
            key = key.replacingOccurrences(of: ".attention.k_lin.", with: ".attention.key_proj.")
            key = key.replacingOccurrences(of: ".attention.v_lin.", with: ".attention.value_proj.")
            key = key.replacingOccurrences(of: ".attention.out_lin.", with: ".attention.out_proj.")
            key = key.replacingOccurrences(of: ".sa_layer_norm.", with: ".ln1.")
            key = key.replacingOccurrences(of: ".ffn.lin1.", with: ".linear1.")
            key = key.replacingOccurrences(of: ".ffn.lin2.", with: ".linear2.")
            key = key.replacingOccurrences(of: ".output_layer_norm.", with: ".ln2.")
            key = key.replacingOccurrences(of: "vocab_transform", with: "lm_head.dense")
            key = key.replacingOccurrences(of: "vocab_layer_norm", with: "lm_head.ln")
            key = key.replacingOccurrences(of: "vocab_projector", with: "lm_head.decoder")
            key = key.replacingOccurrences(of: "distilbert.", with: "")
            result[key] = item.value
        }.filter { key, _ in key != "embeddings.position_ids" }
    }
}

public struct BertConfiguration: Decodable, Sendable {
    var layerNormEps: Float = 1e-12
    var maxTrainedPositions: Int = 2048
    var embedDim: Int = 768
    var numHeads: Int = 12
    var interDim: Int = 3072
    var numLayers: Int = 12
    var typeVocabularySize: Int = 2
    var vocabularySize: Int = 30528
    var maxPositionEmbeddings: Int = 0
    var modelType: String

    enum CodingKeys: String, CodingKey {
        case layerNormEps = "layer_norm_eps"
        case maxTrainedPositions = "max_trained_positions"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
    }

    enum BertCodingKeys: String, CodingKey {
        case embedDim = "hidden_size"
        case numHeads = "num_attention_heads"
        case interDim = "intermediate_size"
        case numLayers = "num_hidden_layers"
        case typeVocabularySize = "type_vocab_size"
    }

    enum DistilBertCodingKeys: String, CodingKey {
        case embedDim = "dim"
        case numLayers = "n_layers"
        case numHeads = "n_heads"
        case interDim = "hidden_dim"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<CodingKeys> =
            try decoder.container(
                keyedBy: CodingKeys.self)
        layerNormEps =
            try container.decodeIfPresent(
                Float.self,
                forKey: CodingKeys.layerNormEps.self)
            ?? 1e-12
        maxTrainedPositions =
            try container.decodeIfPresent(
                Int.self,
                forKey: CodingKeys.maxTrainedPositions
                    .self) ?? 2048
        vocabularySize =
            try container.decodeIfPresent(
                Int.self,
                forKey: CodingKeys.vocabularySize.self)
            ?? 30528
        maxPositionEmbeddings =
            try container.decodeIfPresent(
                Int.self,
                forKey: CodingKeys.maxPositionEmbeddings
                    .self) ?? 0
        modelType = try container.decode(String.self, forKey: CodingKeys.modelType.self)

        if modelType == "distilbert" {
            let distilBertConfig: KeyedDecodingContainer<DistilBertCodingKeys> =
                try decoder.container(
                    keyedBy: DistilBertCodingKeys.self)
            embedDim =
                try distilBertConfig.decodeIfPresent(
                    Int.self,
                    forKey: DistilBertCodingKeys.embedDim.self) ?? 768
            numHeads =
                try distilBertConfig.decodeIfPresent(
                    Int.self,
                    forKey: DistilBertCodingKeys.numHeads.self) ?? 12
            interDim =
                try distilBertConfig.decodeIfPresent(
                    Int.self, forKey: DistilBertCodingKeys.interDim.self)
                ?? 3072
            numLayers =
                try distilBertConfig.decodeIfPresent(
                    Int.self,
                    forKey: DistilBertCodingKeys.numLayers.self) ?? 12
            typeVocabularySize = 0
        } else {
            let bertConfig: KeyedDecodingContainer<BertCodingKeys> = try decoder.container(
                keyedBy: BertCodingKeys.self)

            embedDim =
                try bertConfig.decodeIfPresent(
                    Int.self,
                    forKey: BertCodingKeys.embedDim.self) ?? 768
            numHeads =
                try bertConfig.decodeIfPresent(
                    Int.self,
                    forKey: BertCodingKeys.numHeads.self) ?? 12
            interDim =
                try bertConfig.decodeIfPresent(
                    Int.self, forKey: BertCodingKeys.interDim.self)
                ?? 3072
            numLayers =
                try bertConfig.decodeIfPresent(
                    Int.self,
                    forKey: BertCodingKeys.numLayers.self) ?? 12
            typeVocabularySize =
                try bertConfig.decodeIfPresent(
                    Int.self,
                    forKey: BertCodingKeys.typeVocabularySize
                        .self) ?? 2
        }
    }
}
