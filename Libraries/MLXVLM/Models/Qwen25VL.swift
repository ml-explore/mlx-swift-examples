//
//  Qwen25VL.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 2/1/25.
//

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_5_vl

import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Common

/// Rotates half the hidden dims of the input
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let index = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< index]
    let x2 = x[.ellipsis, index...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Language

private enum Language {
    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            if let v = args.ropeScaling?["mrope_section"], let array = v.asInts() {
                self.mropeSection = sequence(state: (0, array.makeIterator())) { state in
                    if let v = state.1.next() {
                        // note the *2
                        state.0 += v * 2
                        return state.0
                    } else {
                        return nil
                    }
                }.dropLast()
            } else {
                fatalError("rope_scaling['mrope_section'] must be array of integers")
            }

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: args.ropeTraditional,
                base: args.ropeTheta
            )
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray? = nil,
            cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            let mask = mask?[0..., 0 ..< keys.dim(-2)]

            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLDecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen25Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Qwen25VLDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    Qwen25VLDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen25Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self.model = Qwen25Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum Vision {
    static fileprivate func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    fileprivate class VisionRotaryEmbedding: Module {
        let dimensions: Int
        let theta: Float

        init(dimensions: Int, theta: Float = 10000.0) {
            self.dimensions = dimensions
            self.theta = theta
        }

        func callAsFunction(_ sequenceLength: Int) -> MLXArray {
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            let inverseFreq = 1.0 / pow(theta, p)
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            return outer(seq, inverseFreq)
        }
    }

    fileprivate class PatchEmbed: Module {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int = 14,
            temporalPatchSize: Int = 2,
            inChannels: Int = 3,
            hiddenSize: Int = 1152
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: false
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            hiddenStates = proj(hiddenStates)
            hiddenStates = hiddenStates.reshaped(-1, hiddenSize)
            return hiddenStates
        }
    }

    fileprivate class PatchMerger: Module {
        let hiddenSize: Int

        @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int = 2) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = RMSNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int = 16) {
            self.numHeads = numHeads
            let headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
            self._proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            frames: [THW],
            rotaryPositionEmbedding: MLXArray? = nil
        ) -> MLXArray {
            let seqLength = x.dim(0)
            let B = frames[0].t
            let L = seqLength / B

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(seqLength, numHeads, -1)
            k = k.reshaped(seqLength, numHeads, -1)
            v = v.reshaped(seqLength, numHeads, -1)

            if let rotaryPositionEmbedding {
                q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
                k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)
            }

            q = q.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: nil
            )
            .transposed(0, 2, 1, 3)
            .reshaped(seqLength, -1)

            return proj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen25VLVisionBlock: Module {
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dim: config.hiddenSize,
                numHeads: config.numHeads
            )

            self.mlp = MLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            frames: [THW],
            rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    frames: frames,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )

            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]

        let windowSize: Int
        let patchSize: Int
        let spatialMergeSize: Int
        let spatialMergeUnit: Int
        let fullAttBlockIndexes: [Int]

        init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.windowSize = config.windowSize
            self.patchSize = config.patchSize
            self.spatialMergeSize = config.spatialMergeSize
            self.fullAttBlockIndexes = config.fullAttBlockIndexes

            self.spatialMergeUnit = spatialMergeSize * spatialMergeSize

            self._patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize
            )

            let headDim = config.hiddenSize / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = VisionRotaryEmbedding(
                dimensions: headDim / 2
            )

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen25VLVisionBlock(config)
            }

            self._merger.wrappedValue = PatchMerger(
                dimensions: config.outHiddenSize, contextDimensions: config.hiddenSize
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            frames: [THW],
            outputHiddenStates: Bool = false
        ) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            var rotaryPosEmb = getRotaryPosEmb(frames)
            var (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)

            // Assuming grid_thw has shape (batch_size, 3)
            let batchSize = frames.count

            // Window processing
            for (layerNum, block) in blocks.enumerated() {
                hiddenStates = block(
                    hiddenStates,
                    frames: frames,
                    rotaryPositionEmbedding: rotaryPosEmb
                )
            }

            hiddenStates = merger(hiddenStates)
            let reverseIndices = argSort(windowIndex, axis: 0)
            hiddenStates = hiddenStates[reverseIndices, 0...]

            return hiddenStates
        }

        private func getRotaryPosEmb(_ frames: [THW]) -> MLXArray {
            var posIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds = hposIds.reshaped(
                    h / spatialMergeSize,
                    spatialMergeSize,
                    w / spatialMergeSize,
                    spatialMergeSize
                )
                .transposed(0, 2, 1, 3)
                .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds = wposIds.reshaped(
                    h / spatialMergeSize,
                    spatialMergeSize,
                    w / spatialMergeSize,
                    spatialMergeSize
                )
                .transposed(0, 2, 1, 3)
                .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                posIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(posIds, axis: 0)
            let maxFrameSize = frames.lazy.map({ max($0.h, $0.w) }).max() ?? 0
            let rotaryPosEmb = rotaryPositionEmbedding(maxFrameSize)
            let rotaryPosEmbFull = rotaryPosEmb[indices]

            return rotaryPosEmbFull.reshaped(indices.dim(0), -1)
        }

        private func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexId = [0]
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for row in frames {
                let (gridT, gridH, gridW) = row.values
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize

                // Create initial index array
                let index = MLXArray(0 ..< (gridT * llmGridH * llmGridW)).reshaped(
                    gridT, llmGridH, llmGridW)

                // Calculate padding and window dimensions
                let padH = vitMergerWindowSize - llmGridH % vitMergerWindowSize
                let padW = vitMergerWindowSize - llmGridW % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize

                var indexPadded = padded(
                    index,
                    widths: [0, .init((0, padH)), .init((0, padW))],
                    mode: .constant,
                    value: MLXArray(-100, dtype: index.dtype))

                // Reshape and transpose for window creation
                indexPadded = indexPadded.reshaped(
                    gridT,
                    numWindowsH,
                    vitMergerWindowSize,
                    numWindowsW,
                    vitMergerWindowSize
                )

                indexPadded =
                    indexPadded
                    .transposed(0, 1, 3, 2, 4)
                    .reshaped(
                        gridT,
                        numWindowsH * numWindowsW,
                        vitMergerWindowSize,
                        vitMergerWindowSize
                    )

                // Process sequence lengths and indices
                let seqlens = sum(indexPadded .!= -100, axes: [2, 3]).reshaped(-1)
                indexPadded = indexPadded.reshaped(-1)

                var indices = [Int]()
                for (i, v) in indexPadded.asArray(Int.self).enumerated() {
                    if v != -100 {
                        indices.append(v)
                    }
                }

                let indexNew = MLXArray(indices)

                // Update window index and cumulative sequence lengths
                windowIndex.append(indexNew + windowIndexId)
                let cuSeqlensTmp =
                    cumsum(seqlens, axis: 0) * spatialMergeUnit + (cuWindowSeqlens.last ?? 0)

                cuWindowSeqlens.append(contentsOf: cuSeqlensTmp.asArray(Int.self))
                windowIndexId += [gridT * llmGridH * llmGridW]
            }

            // Create final arrays
            let finalWindowIndex = concatenated(windowIndex, axis: 0)
            let finalCuWindowSeqlens = MLXArray(cuWindowSeqlens)
            return (finalWindowIndex, finalCuWindowSeqlens)
        }

        private func getCuSeqlens(_ gridThw: [THW]) -> MLXArray {
            var cuSeqlens = [MLXArray]()

            // Calculate cumulative sequence lengths for each item in batch
            for row in gridThw {
                let seqLen = row.h * row.w
                let repeatedLen = repeated(MLXArray(seqLen), count: row.t, axis: 0)
                cuSeqlens.append(repeatedLen)
            }

            // Concatenate and process all sequence lengths
            var result = concatenated(cuSeqlens, axis: 0)
            result = cumsum(result.asType(.int32), axis: 0)

            var r = padded(result, width: .init((1, 0)))

            // Add leading zero for offset calculation
            return r
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4 && array.ndim != 5 {
                return false
            }

            if array.dim(-1) == 3 {
                return true
            }

            let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [B, out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [B, out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

public class Qwen25VL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen25VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: Qwen25VLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        // Get input embeddings from language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get hidden states from vision model
        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Insert special image tokens in the input_ids
        return mergeInputIdsWithImageFeatures(
            inputIds: inputIds,
            inputEmbeds: inputEmbeds,
            imageFeatures: hiddenStates
        )
    }

    private func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray,
        inputEmbeds: MLXArray,
        imageFeatures: MLXArray
    ) -> MLXArray {
        let imageTokenIndex = config.baseConfiguration.imageTokenId
        let videoTokenIndex = config.baseConfiguration.videoTokenId

        var imageIndices = [Int]()
        for (i, v) in inputIds.asArray(Int.self).enumerated() {
            if v == imageTokenIndex || v == videoTokenIndex {
                imageIndices.append(i)
            }
        }

        // Make sure shapes match before assignment
        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }

        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures[.newAxis, 0..., 0...]
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        if let videoPixels = input.video?.pixels, let videoFrames = input.video?.frames {
            if allPixels == nil {
                allPixels = videoPixels.asType(dtype)
            } else {
                allPixels = concatenated([allPixels!, videoPixels.asType(dtype)])
            }
            allFrames.append(contentsOf: videoFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionModel.sanitize(
            weights: Dictionary(
                uniqueKeysWithValues: weights.map { key, value in
                    var key = key
                    if !key.contains("vision_tower") {
                        key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                    }
                    if !key.contains("language_model") {
                        key = key.replacingOccurrences(of: "model", with: "language_model.model")
                        key = key.replacingOccurrences(
                            of: "lm_head", with: "language_model.lm_head")
                    }
                    return (key, value)
                })
        )
    }
}

// MARK: - Configuration

public struct Qwen25VLConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int { _maxPositionEmbeddings ?? 128000 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let ropeScaling: [String: StringOrNumber]?
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        public let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullAttBlockIndexes: [Int]
        public let tokensPerSecond: Int

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case _inChannels = "in_channels"
            case _layerNormEps = "layer_norm_eps"
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size"
            case fullAttBlockIndexes = "fullatt_block_indexes"
            case tokensPerSecond = "tokens_per_second"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let videoTokenId: Int
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_id"
            case videoTokenId = "video_token_id"
            case hiddenSize = "hidden_size"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Vision config is in a sub-dictionary
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self,
            forKey: .visionConfiguration
        )

        // Text and base configs are overlaid in the top level
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

// MARK: - Processor

/// Qwen25VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen25VL`` and is typically created by ``VLMModelFactory``.
///
public typealias Qwen25VLProcessor = QwenVLProcessor<Qwen25VLProcessorConfiguration>

// Configuration for ``Qwen25VLProcessor``
public struct Qwen25VLProcessorConfiguration: QwenVLProcessorConfiguration {
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let maxPixels: Int
    public let minPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case maxPixels = "max_pixels"
        case minPixels = "min_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
    }

    private var chatTemplate: String {
        "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    }

    public func applyChatTemplate(messages: [Message], tokenizer: any Tokenizer) throws -> [Int] {
        return try tokenizer.applyChatTemplate(messages: messages, chatTemplate: chatTemplate)
    }
}
