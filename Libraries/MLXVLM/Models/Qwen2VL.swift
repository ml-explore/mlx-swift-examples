// Copyright © 2024 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_vl

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

    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
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

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
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
                // mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
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
                fatalError("rope_scaling['mrope_section'] must be an array of integers")
            }

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
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
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

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

    fileprivate class Qwen2VLDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
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

    fileprivate class Qwen2Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Qwen2VLDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    Qwen2VLDecoderLayer(args)
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
        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
            self.model = Qwen2Model(args)

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

    fileprivate class VisionRotaryEmbedding {
        let dimensions: Int
        let theta: Float
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float) {
            self.dimensions = dimensions
            self.theta = theta
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            self.inverseFreq = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            let freqs = outer(seq, inverseFreq)
            return freqs
        }
    }

    fileprivate class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let embedDimensions: Int

        init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, embedDimensions: Int) {
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

    fileprivate class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: LayerNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = LayerNorm(dimensions: contextDimensions, eps: 1e-6)
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

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, frames: [THW], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)
            let B = frames[0].t
            let L = sequenceLength / B

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: nil
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo var activation: GELU
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self.activation = GELU(approximation: .fast)
            self.fc1 = Linear(dimensions, hiddenDimensions)
            self.fc2 = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    fileprivate class Qwen2VLVisionBlock: Module {

        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Qwen2VLConfiguration.VisionConfiguration) {
            self.norm1 = LayerNorm(dimensions: config.embedDimensions, eps: 1e-6)
            self.norm2 = LayerNorm(dimensions: config.embedDimensions, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dims: config.embedDimensions, numHeads: config.numHeads)

            let mlpHiddenDimensions = Int(Float(config.embedDimensions) * config.mlpRatio)
            self.mlp = MLP(
                dimensions: config.embedDimensions, hiddenDimensions: mlpHiddenDimensions)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, frames: [THW], rotaryPositionEmbedding: MLXArray
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
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen2VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: PatchMerger

        let spatialMergeSize: Int

        public init(_ config: Qwen2VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize

            self._patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                embedDimensions: config.embedDimensions)

            let headDimensions = config.embedDimensions / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = VisionRotaryEmbedding(
                dimensions: headDimensions / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen2VLVisionBlock(config)
            }
            self._patchMerger.wrappedValue = PatchMerger(
                dimensions: config.hiddenSize, contextDimensions: config.embedDimensions,
                spatialMergeSize: 2)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[
                indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotaryPositionEmbedding = rotaryPositionEmbedding(frames)

            let batchSize = frames.count

            for block in blocks {
                hiddenStates = block(
                    hiddenStates, frames: frames,
                    rotaryPositionEmbedding: rotaryPositionEmbedding)
            }

            return patchMerger(hiddenStates)
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4, array.ndim != 5 {
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

// MARK: - Processor

/// Qwen2VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen2VL`` and is typically created by ``VLMModelFactory``.
public class Qwen2VLProcessor: UserInputProcessor {
    private let config: Qwen2VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen2VLProcessorConfiguration, tokenizer: any Tokenizer) {
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

        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)

        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
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

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = input.prompt.asMessages()
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })
            if let imageFrames = processedImage?.frames {
                promptTokens = try replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>")
            }
        }

        // Process videos if any
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[CIImage]]()
            for video in input.videos {
                if let imageSequence = try? await MediaProcessing.asCIImageSequence(
                    video.asAVAsset(), samplesPerSecond: 2)
                {
                    videosAsImageSequences.append(imageSequence)
                }
            }
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try preprocess(images: $0, processing: input.processing)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>")
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }

    func replacePaddingTokens(in promptTokens: [Int], frames: [THW], paddingToken: String)
        throws -> [Int]
    {
        // Replace single padding token with correct number for each image or video frame
        let placeholderTokens = try tokenizer.encode(
            text: "<|vision_start|>\(paddingToken)<|vision_end|>")
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens does not match number of frames")
        }
        let mergeLength = config.mergeSize * config.mergeSize
        let replacementSequences = try frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return try tokenizer.encode(
                text:
                    "<|vision_start|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|vision_end|>"
            )
        }
        // Build the final array
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            // Add tokens before the placeholder
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            // Add replacement sequence
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        // Add any remaining tokens after the last replacement
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }
}

// MARK: - Model

/// Qwen2VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class Qwen2VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen2VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: Qwen2VLConfiguration) {
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

        // Get the input embeddings from the language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get the ouptut hidden states from the vision model
        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Insert special image tokens in the input_ids
        return mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates)
    }

    private func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray
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

        // Process both images and videos together
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
            weights:
                Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        var key = key
                        if !key.contains("vision_tower") {
                            key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                        }
                        if !key.contains("language_model") {
                            key = key.replacingOccurrences(
                                of: "model", with: "language_model.model")
                            key = key.replacingOccurrences(
                                of: "lm_head", with: "language_model.lm_head")
                        }

                        return (key, value)
                    })
        )
    }

}

// MARK: - Configuration

/// Configuration for ``Qwen2VL``
public struct Qwen2VLConfiguration: Codable, Sendable {

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
        public var maxpPositionEmbeddings: Int { _maxPositionEmbeddings ?? 32768 }
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
        public let embedDimensions: Int
        public let hiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let mlpRatio: Float
        public let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        public let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int

        enum CodingKeys: String, CodingKey {
            case depth
            case embedDimensions = "embed_dim"
            case hiddenSize = "hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case mlpRatio = "mlp_ratio"
            case _inChannels = "in_channels"
            case _layerNormEps = "layer_norm_eps"
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
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

        // this is a sub-dictionary
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // these are overlaid in the top level
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``Qwen2VLProcessor``
public struct Qwen2VLProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    private let _size: Size?
    private let _maxPixels: Int?
    private let _minPixels: Int?

    public var minPixels: Int {
        _minPixels ?? _size?.minPixels ?? 3136
    }
    public var maxPixels: Int {
        _maxPixels ?? _size?.maxPixels ?? 12_845_056
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case _maxPixels = "max_pixels"
        case _minPixels = "min_pixels"
        case _size = "size"
    }
}
