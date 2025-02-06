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
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float = 10000.0) {
            self.dimensions = dimensions
            self.theta = theta
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            self.inverseFreq = 1.0 / pow(theta, p)
        }

        func callAsFunction(_ sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            return outer(seq, inverseFreq)
        }
    }

    fileprivate class PatchEmbed: Module {
        @ModuleInfo var proj: Conv3d
        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let embedDimensions: Int

        init(
            patchSize: Int = 14,
            temporalPatchSize: Int = 2,
            inChannels: Int = 3,
            embedDimensions: Int = 1152
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.embedDimensions = embedDimensions

            let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: embedDimensions,
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
            hiddenStates = hiddenStates.reshaped(-1, embedDimensions)
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
            cuSeqlens: MLXArray,
            rotaryPositionEmbedding: MLXArray? = nil
        ) -> MLXArray {
            let seqLength = x.dim(0)
            let qkv = qkv(x).reshaped(seqLength, 3, numHeads, -1).transposed(1, 0, 2, 3)
            let (q, k, v) = (
                qkv[0].expandedDimensions(axis: 0), qkv[1].expandedDimensions(axis: 0),
                qkv[2].expandedDimensions(axis: 0)
            )

            var queries = q
            var keys = k

            if let rotaryPositionEmbedding {
                queries = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
                keys = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries.transposed(0, 2, 1, 3),
                keys: keys.transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3),
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
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
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
            cuSeqlens: MLXArray,
            rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
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
                embedDimensions: config.hiddenSize
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
            gridThw: [THW],
            outputHiddenStates: Bool = false
        ) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            var rotaryPosEmb = getRotaryPosEmb(gridThw)
            var (windowIndex, cuWindowSeqlens) = getWindowIndex(gridThw)

            let seqlensArray = cuWindowSeqlens.asArray(Int.self)
            var seen = Set<Int>()
            var idx: [Int32] = []

            for (i, x) in seqlensArray.enumerated() {
                if !seen.contains(x) {
                    seen.insert(x)
                    idx.append(Int32(i))
                }
            }

            let idx1 = MLXArray(idx)
            cuWindowSeqlens = cuWindowSeqlens[idx1]

            let seqLen = hiddenStates.dim(0)
            hiddenStates = hiddenStates.reshaped(seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            hiddenStates = hiddenStates[windowIndex, 0..., 0...]
            hiddenStates = hiddenStates.reshaped(seqLen, -1)

            rotaryPosEmb = rotaryPosEmb.reshaped(seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            rotaryPosEmb = rotaryPosEmb[windowIndex, 0..., 0...]
            rotaryPosEmb = rotaryPosEmb.reshaped(seqLen, -1)

            // Assuming grid_thw has shape (batch_size, 3)
            let batchSize = gridThw.count

            var cuSeqlens: [MLXArray] = []
            for row in gridThw {
                let (gridT, gridH, gridW) = row.values
                let seqLen = gridH * gridW
                let repeats = gridT

                // Create array with repeated values
                let repeatedSeq = MLXArray.full([repeats], values: MLXArray(seqLen))

                cuSeqlens.append(repeatedSeq)
            }

            let cuSeqlensPadded = padded(
                cumsum(concatenated(cuSeqlens)),
                width: .init((1, 0)),
                mode: .constant,
                value: MLXArray(0)
            )

            // Window processing
            for (layerNum, block) in blocks.enumerated() {
                let cuSeqlensNow =
                    fullAttBlockIndexes.contains(layerNum) ? cuSeqlensPadded : cuWindowSeqlens
                hiddenStates = block(
                    hiddenStates,
                    cuSeqlens: cuSeqlensNow,
                    rotaryPositionEmbedding: rotaryPosEmb
                )
            }

            hiddenStates = merger(hiddenStates)
            let reverseIndices = argSort(windowIndex, axis: 0)
            hiddenStates = hiddenStates[reverseIndices, 0...]

            return hiddenStates
        }

        private func getRotaryPosEmb(_ gridThw: [THW]) -> MLXArray {
            var posIds = [MLXArray]()

            for row in gridThw {
                let (t, h, w) = row.values

                // Create and process horizontal position IDs
                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds = hposIds.reshaped(
                    h / spatialMergeSize,
                    spatialMergeSize,
                    w / spatialMergeSize,
                    spatialMergeSize
                )
                hposIds = hposIds.transposed(0, 2, 1, 3)
                hposIds = hposIds.flattened()

                // Create and process vertical position IDs
                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds = wposIds.reshaped(
                    h / spatialMergeSize,
                    spatialMergeSize,
                    w / spatialMergeSize,
                    spatialMergeSize
                )
                wposIds = wposIds.transposed(0, 2, 1, 3)
                wposIds = wposIds.flattened()

                // Stack and tile position IDs
                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                posIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(posIds, axis: 0)
            let maxGridSize = gridThw.lazy.map({ max($0.h, $0.w) }).max() ?? 0
            let rotaryPosEmbFull = rotaryPositionEmbedding(maxGridSize)
            let rotaryPosEmb = rotaryPosEmbFull[indices]

            return rotaryPosEmb.reshaped(indices.dim(0), -1)
        }

        private func getWindowIndex(_ gridThw: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexId = [0]
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for row in gridThw {
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

                indexPadded = indexPadded.transposed(0, 1, 3, 2, 4)
                indexPadded = indexPadded.reshaped(
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

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, gridThw: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let gridThw else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        // Get input embeddings from language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get hidden states from vision model
        var hiddenStates = self.visionModel(pixelValues, gridThw: gridThw)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Merge input IDs with image features
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
            if v == imageTokenIndex {
                imageIndices.append(i)
            }
        }

        if imageIndices.isEmpty {
            for (i, v) in inputIds.asArray(Int.self).enumerated() {
                if v == videoTokenIndex {
                    imageIndices.append(i)
                }
            }
        }

        inputEmbeds[0..., MLXArray(imageIndices), 0...] = imageFeatures
        return inputEmbeds
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        let imageGridThw = input.image?.imageGridThw
        let imagePixels = input.image?.pixels.asType(dtype)

        let videoGridThw = input.video?.videoGridThw
        let videoPixels = input.video?.pixels.asType(dtype)

        let gridThw: [THW]?
        let pixels: MLXArray?

        if videoGridThw == nil {
            gridThw = imageGridThw
            pixels = imagePixels
        } else {
            gridThw = videoGridThw
            pixels = videoPixels
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: pixels,
            gridThw: gridThw
        )

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

public class Qwen25VLProcessor: UserInputProcessor {
    private let config: Qwen25VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen25VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    // image_processing_qwen2_vl.smart_resize
    private func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws -> (Int, Int)
    {
        if height < factor {
            throw VLMError.imageProcessingFailure(
                "height: \(height) must be larger than factor: \(factor)")
        }
        if width < factor {
            throw VLMError.imageProcessingFailure(
                "width: \(width) must be larger than factor: \(factor)")
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "absolute aspect ratio must be smaller than 200: \(width)x\(height)")
        }

        var hBar = Int(round(Float(height) / Float(factor))) * factor
        var wBar = Int(round(Float(width) / Float(factor))) * factor

        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(floor(Float(height) * beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) * beta / Float(factor))) * factor
        }
        return (hBar, wBar)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // first apply the user requested resizing, etc. if any
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        // image_processing_qwen2_vl._preprocess

        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels, maxPixels: config.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages =
            try images
            .map {
                MediaProcessing.inSRGBToneCurveSpace($0)
            }
            .map {
                return MediaProcessing.resampleBicubic($0, to: resizedSize)
            }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple)
            }
            .map {
                MediaProcessing.asMLXArray($0)
            }

        var patches = concatenated(processedImages)
        let mod = patches.dim(0) % config.temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let lastPatchRepeated = tiled(
                lastPatch, repetitions: [config.temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, lastPatchRepeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / self.config.temporalPatchSize
        let gridH = resizedHeight / self.config.patchSize
        let gridW = resizedWidth / self.config.patchSize

        patches = patches.reshaped(
            gridT,
            config.temporalPatchSize,
            channel,
            gridH / config.mergeSize,
            config.mergeSize,
            config.patchSize,
            gridW / config.mergeSize,
            config.mergeSize,
            config.patchSize
        )
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        let flattenedPatches = patches.reshaped(
            gridT * gridH * gridW,
            channel * config.temporalPatchSize * config.patchSize * config.patchSize
        )

        return (flattenedPatches, .init(gridT, gridH, gridW))
    }

    public func prepare(prompt: UserInput.Prompt, imageTHW: [THW]?, videoTHW: [THW]?) -> String {
        // the tokenizer does have a chat template and it expects messages
        // like this:
        //
        // [{'role': 'user', 'content': [{'type': 'text', 'text': 'What are these?'},
        //  {'type': 'image'}, {'type': 'image'}, {'type': 'image'}]}]
        //
        // The output of the prompt template is fed into
        // image_processing_qwen2_vl.preprocess where it is further augmented
        // by replacing tokens according to imageTHW.
        //
        // Neither the structured content nor the postprocessing of the template
        // are supported in current Tokenizer/Jinja (swift) so handle that here.

        var messages = prompt.asMessages()
        if messages[0]["role"] != "system" {
            messages.insert(["role": "system", "content": "You are a helpful assistant."], at: 0)
        }

        let lastIndex = messages.count - 1
        var lastMessage = messages[lastIndex]["content"] ?? ""

        // image_processing_qwen2_vl.preprocess -- inject image_pad tokens for each image
        let mergeLength = config.mergeSize * config.mergeSize
        for thw in imageTHW ?? [] {
            lastMessage += "<|vision_start|>"
            lastMessage += Array(repeating: "<|image_pad|>", count: thw.product / mergeLength)
                .joined()
            lastMessage += "<|vision_end|>"
        }

        for thw in videoTHW ?? [] {
            lastMessage += "<|vision_start|>"
            lastMessage += Array(repeating: "<|video_pad|>", count: thw.product / mergeLength)
                .joined()
            lastMessage += "<|vision_end|>"
        }

        messages[lastIndex]["content"] = lastMessage

        return
            messages
            .map {
                "<|im_start|>\($0["role"] ?? "user")\n\($0["content"] ?? "")<|im_end|>"
            }
            .joined(separator: "\n")
            + "\n<|im_start|>assistant\n"
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        if input.images.isEmpty && input.videos.isEmpty {
            // just a straight text prompt
            let prompt = prepare(prompt: input.prompt, imageTHW: nil, videoTHW: nil)
            let promptTokens = try tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // image_processing_qwen2_vl.preprocess
        let images = try input.images.map {
            try preprocess(images: [$0.asCIImage()], processing: input.processing)
        }

        var videosAsImageSequences = [[CIImage]]()
        for video in input.videos {
            if let imageSequence = try? await MediaProcessing.asCIImageSequence(
                video.asAVAsset(), samplesPerSecond: 2)
            {
                videosAsImageSequences.append(imageSequence)
            }
        }
        let videos = try videosAsImageSequences.map {
            try preprocess(images: $0, processing: input.processing)
        }

        let imagePixels: MLXArray?
        let image: LMInput.ProcessedImage?
        if !images.isEmpty {
            imagePixels = concatenated(images.map { $0.0 })
            image = LMInput.ProcessedImage(pixels: imagePixels!, imageGridThw: images.map { $0.1 })
        } else {
            imagePixels = nil
            image = nil
        }

        let videoPixels: MLXArray?
        let video: LMInput.ProcessedVideo?
        if !videos.isEmpty {
            videoPixels = concatenated(videos.map { $0.0 })
            video = LMInput.ProcessedVideo(pixels: videoPixels!, videoGridThw: videos.map { $0.1 })
        } else {
            videoPixels = nil
            video = nil
        }

        let prompt = prepare(
            prompt: input.prompt, imageTHW: image?.imageGridThw, videoTHW: video?.videoGridThw)
        let promptTokens = try tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: image, video: video)
    }
}

public struct Qwen25VLProcessorConfiguration: Codable, Sendable {

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let maxPixels: Int
    public let minPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case maxPixels = "max_pixels"
        case minPixels = "min_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
    }
}
