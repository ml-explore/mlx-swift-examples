// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen3VLLanguage {


    final class RotaryEmbedding {

        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(headDim: Int, base: Double, ropeScaling: Qwen3VLConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            self.invFreq = 1.0 / pow(baseArray, freq)
            self.mropeSection = ropeScaling?.mropeSection ?? [24, 20, 20]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqs_t = freqs[0, 0..., 0..., 0...]  // (bs, seq_len, head_dim // 2)
            
            let dims = freqs_t.dim(-1)
            var slices: [MLXArray] = []
            
            for idx in 0..<dims {
                var slice = freqs_t[0..., 0..., idx]
                
                for (dimIndex, offset) in [(1, 1), (2, 2)] {
                    let end = min(mropeSection[dimIndex] * 3, dims)
                    if idx >= offset && idx < end && (idx - offset) % 3 == 0 {
                        slice = freqs[dimIndex, 0..., 0..., idx]
                        break
                    }
                }
                
                slices.append(slice)
            }
            
            return stacked(slices, axis: -1)
        }

        func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = positionIds[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds, repetitions: [3, 1, 1])
            }

            let pos = positionIds.asType(.float32)
            var invFreq = self.invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            var emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }


    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)
            
            
            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)
            
            return result
        }
    }


    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let visualMask
                {
                    
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            let indices = maskIndices(visualMask)
            guard !indices.isEmpty else { return hiddenStates }
            
            let indexArray = MLXArray(indices.map { UInt32($0) })
            
            var result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds
            
            return result
        }

        private func maskIndices(_ mask: MLXArray) -> [Int] {
            let bools = mask.asType(.bool).asArray(Bool.self)
            var indices: [Int] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(idx)
            }
            return indices
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLConfiguration
        let textConfig: Qwen3VLConfiguration.TextConfiguration
        var kvHeads: [Int]

        private var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(repeating: config.textConfiguration.numKeyValueHeads,
                                 count: config.textConfiguration.numHiddenLayers)

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false)
            }
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            if pixelValues != nil {
                ropeDeltas = nil
            }

            var positionIds = providedPositionIds
            
            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                if (cache?.first?.offset ?? 0) == 0 || ropeDeltas == nil || cache == nil {
                    if let inputIds {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)
                        
                        positionIds = computed
                        ropeDeltas = deltas
                    } else if let cache, ropeDeltas == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0
                        
                        var base = MLXArray(0..<seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue
                        
                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
                    }
                } else if let cache, let ropeDeltas {
                    let batch = (inputIds ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIds ?? inputEmbeddings!).dim(1)
                    
                    let lastCacheOffset = cache.last?.offset ?? 0
                    
                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)
                    
                    var base = MLXArray(0..<seqLength).asType(.int32)
                    base = base[.newAxis, 0...]
                    base = broadcast(base, to: [batch, seqLength])
                    
                    if delta.dim(0) == 1 && batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }
                    
                    base = base + delta
                    
                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = broadcast(positionIds!, to: [3, batch, seqLength])                    
                }
            }
            
            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output)
        }

    }
}
