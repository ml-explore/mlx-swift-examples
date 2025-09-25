import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

public class EmbeddingGemma: Module, EmbeddingModel {
    @ModuleInfo private var model: Gemma3TextModel
    @ModuleInfo private var dense: [Module]

    public let config: Gemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = Gemma3TextModel(config)
        self.dense = [
            Linear(768, 3072, bias: false), Linear(3072, 768, bias: false)
        ]
    }

    public func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        var out = model.getHiddenStates(inputs, mask: nil, cache: nil)

        // mean pooling
        let notPadding = inputs .!= 0
        let sum = (out * notPadding[.ellipsis, .newAxis]).sum(axis:1)
        let nonMasked = notPadding.sum(axis: -1, keepDims: true)
        out = sum / nonMasked

        for dense in self.dense {
            if let dense = dense as? Linear {
                out = dense(out)
            } else if let dense = dense as? QuantizedLinear {
                out = dense(out)
            }
        }

        // normalize
        out = out.asType(Float32.self)
        let norm = maximum(norm(out, ord:2.0, axis:-1, keepDims: true), MLXArray(1e-6))
        let pooledOutput = out / norm

        return EmbeddingModelOutput(hiddenStates: out, pooledOutput: pooledOutput)
    }
    
    /// Get hidden states before the dense projection head
    public func getHiddenStates(_ inputs: MLXArray,  mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil, cache: [KVCache]? = nil) -> MLXArray {
        return model(inputs, mask: mask, cache: cache)
    }


    public func sanitize(weights: [String: MLXArray], quantizationConfig: MLXLMCommon.BaseConfiguration.Quantization? = nil)
        -> [String: MLXArray]
    {
        var processedWeights = model.sanitize(weights: weights, quantizationConfig: quantizationConfig)

        // 1. Add a model. prefix to all model. weights
        processedWeights = Dictionary(uniqueKeysWithValues: processedWeights.map { key, value in
            if key.hasPrefix("model.") || key.hasPrefix("lm_head.") {
                return ("model.\(key)", value)
            } else {
                return (key, value)
            }
        })

        // 2. Apply quantization to dense layers, if needed
        let hasQuantizedDense = hasQuantizedWeights(layerPath: "dense.0", in: processedWeights)
        if hasQuantizedDense {
            let groupSize = quantizationConfig?.groupSize ?? 64
            let bits = quantizationConfig?.bits ?? 4

            quantize(model: self) { path, module in
                if hasQuantizedWeights(layerPath: path, in: processedWeights) {
                    return (groupSize, bits)
                }
                return nil
            }
        }

        return processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    public func sanitize(weights: [String : MLXArray]) -> [String : MLXArray] {
        sanitize(weights: weights, quantizationConfig: nil)
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
}
