#  MLXEmbedders

This directory contains ports of popular Encoders / Embedding Models. 

## Usage Example

```swift

let modelContainer = try await MLXEmbedders.loadModelContainer(
    configuration: ModelConfiguration.nomic_text_v1_5)
let result = await modelContainer.perform {
    (model: EmbeddingModel, tokenizer, pooling) -> [[Float]] in
    let inputs = [
        "search_query: Animals in Tropical Climates.",
        "search_document: Elephants",
        "search_document: Horses",
        "search_document: Polar Bears",
    ].map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
    }
    // Pad to longest
    let maxLength = inputs.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
    }

    let padded = stacked(
        inputs.map { elem in
            MLXArray(
                elem
                    + Array(
                        repeating: tokenizer.eosTokenId ?? 0,
                        count: maxLength - elem.count))
        })
    let mask = (padded .!= tokenizer.eosTokenId ?? 0)
    let tokenTypes = MLXArray.zeros(like: padded)
    let result = pooling(
        model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true, applyLayerNorm: true
    ).eval()
    return result.map { $0.asArray(Float.self) }
}
```


Ported to swift from [taylorai/mlx_embedding_models](https://github.com/taylorai/mlx_embedding_models/tree/main)
