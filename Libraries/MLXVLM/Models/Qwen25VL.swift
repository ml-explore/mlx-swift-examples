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

// MARK: - Language

private enum Language {
    fileprivate class Attention: QwenVLLanguage.Attention {
        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            super.init(
                hiddenSize: args.hiddenSize,
                attentionHeads: args.attentionHeads,
                kvHeads: args.kvHeads,
                ropeTheta: args.ropeTheta,
                ropeTraditional: args.ropeTraditional,
                ropeScaling: args.ropeScaling
            )
        }
    }

    fileprivate class Qwen25VLDecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: QwenVLLanguage.MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen25VLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = QwenVLLanguage.MLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
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
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: QwenVLVision.Attention
        @ModuleInfo var mlp: MLP

        init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.norm1 = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self._attention.wrappedValue = QwenVLVision.Attention(
                dims: config.hiddenSize,
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
        @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVLVision.PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding:
            QwenVLVision.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: QwenVLVision.PatchMerger

        let spatialMergeSize: Int

        init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize
            self._patchEmbed.wrappedValue = QwenVLVision.PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                embedDimensions: config.hiddenSize
            )

            let headDim = config.hiddenSize / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = QwenVLVision.VisionRotaryEmbedding(
                dimensions: headDim / 2
            )

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen25VLVisionBlock(config)
            }

            self._patchMerger.wrappedValue = QwenVLVision.PatchMerger(
                dimensions: config.outHiddenSize, contextDimensions: config.hiddenSize
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            frames: [THW],
            outputHiddenStates: Bool = false
        ) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            var rotaryPositionEmbedding = rotaryPositionEmbedding(frames)

            let batchSize = frames.count
            for block in blocks {
                hiddenStates = block(
                    hiddenStates, frames: frames,
                    rotaryPositionEmbedding: rotaryPositionEmbedding)
            }
            return patchMerger(hiddenStates)
        }

        private func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

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
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map({ max($0.h, $0.w) }).max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(maxFrameSize)[indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
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
    }
}

// MARK: - Main Model

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
            imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId
        )
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

// MARK: - Processor

/// Qwen25VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen25VL`` and is typically created by ``VLMModelFactory``.
///
public typealias Qwen25VLProcessor = QwenVLProcessor<Qwen25VLProcessorConfiguration>

// MARK: - Configuration

/// Configuration for ``Qwen25VL``
public struct Qwen25VLConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable, QwenVLTextConfigurable {
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

    public struct VisionConfiguration: Codable, Sendable, QwenVLVisionConfigurable {
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

    public struct BaseConfiguration: Codable, Sendable, QwenVLBaseConfiguration {
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

// MARK: - Processor Configuration

/// Configuration for ``Qwen25VLProcessor``
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
}
