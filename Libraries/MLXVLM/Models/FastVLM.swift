//
//  FastVLM.swift
//  mlx-swift-examples
//
//  Created by Pedro Cuenca on 18/10/25.
//

// Based on https://github.com/Blaizzy/mlx-vlm/pull/502

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration

/// Configuration for ``FastVLM``
public struct FastVLMConfiguration: Codable, Sendable {

    public typealias TextConfiguration = Qwen2VLConfiguration.TextConfiguration

    public struct VisionConfiguration: Codable, Sendable {
        public let classHeadRatio: Float
        public let downPatchSize: Int
        public let downStride: Int
        public let downSamples: [Bool]
        public let embedDimensions: [Int]
        public let hiddenSize: Int
        public let imageSize: Int
        public let intermediateSize: Int
        public let layers: [Int]
        public let layerScaleInitValue: Float
        public let mlpRatios: [Int]
        public let numClasses: Int
        public let patchSize: Int
        public let posEmbedShapes: [[Int]?]
        public let projectionDim: Int
        public let repMixerKernelSize: Int
        public let tokenMixers: [String]

        enum CodingKeys: String, CodingKey {
            case classHeadRatio = "cls_ratio"
            case downPatchSize = "down_patch_size"
            case downStride = "down_stride"
            case downSamples = "downsamples"
            case embedDimensions = "embed_dims"
            case hiddenSize = "hidden_size"
            case imageSize = "image_size"
            case intermediateSize = "intermediate_size"
            case layers
            case layerScaleInitValue = "layer_scale_init_value"
            case mlpRatios = "mlp_ratios"
            case numClasses = "num_classes"
            case patchSize = "patch_size"
            case posEmbedShapes = "pos_embs_shapes"
            case projectionDim = "projection_dim"
            case repMixerKernelSize = "repmixer_kernel_size"
            case tokenMixers = "token_mixers"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        private let _imageTokenIndex: Int?
        public var imageTokenIndex: Int { _imageTokenIndex ?? -200 }
        public let eosTokenId: Int
        public let multimodalProjectorType: String
        public let multimodalProjectorHiddenSize: Int
        public let tokenizerModelMaxLangth: Int
        public let tokenizerPaddingSide: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _imageTokenIndex = "image_token_index"
            case eosTokenId = "eos_token_id"
            case multimodalProjectorType = "mm_projector_type"
            case multimodalProjectorHiddenSize = "mm_hidden_size"
            case tokenizerModelMaxLangth = "tokenizer_model_max_length"
            case tokenizerPaddingSide = "tokenizer_padding_side"
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

// MARK: - Language

/// Adapted from Qwen2VL.Language, without multimodal rotaty position embeddings
private enum Language {

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
            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            let maskConverted: MLXFast.ScaledDotProductAttentionMaskMode =
                if let mask {
                    .array(mask[.ellipsis, 0 ..< keys.dim(-2)])
                } else {
                    .none
                }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: maskConverted
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

            let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

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

    /// Multi-headed Self Attention module
    fileprivate class MHSA: Module {
        let headDim: Int
        let numHeads: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo(key: "attn_drop") var attnDrop: Dropout
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "proj_drop") var projDrop: Dropout

        public init(
            dim: Int, headDim: Int = 32, qkvBias: Bool = false, attnDrop: Float = 0.0,
            projDrop: Float = 0.0
        ) {
            precondition(dim % headDim == 0, "dim should be divisible by headDim")
            self.headDim = headDim
            self.numHeads = dim / headDim
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
            self._attnDrop.wrappedValue = Dropout(p: attnDrop)
            self._proj.wrappedValue = Linear(dim, dim)
            self._projDrop.wrappedValue = Dropout(p: projDrop)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Source: https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/multimodal_encoder/mobileclip/mci.py#L661
            var x = x.transposed(0, 3, 1, 2)
            let (B, C, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let N = H * W

            x = x.flattened(start: 2).transposed(0, 2, 1)  // (B, N, C)
            let qkv = self.qkv(x)
                .reshaped(B, N, 3, numHeads, headDim)
                .transposed(2, 0, 3, 1, 4)

            let q = qkv[0]
            let k = qkv[1]
            let v = qkv[2]

            x = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
            x = x.transposed(0, 2, 1, 3).reshaped(B, N, C)
            x = proj(x)
            x = projDrop(x)

            x = x.reshaped(B, H, W, C)
            return x
        }
    }

    /// Convolutional FFN Module
    fileprivate class ConvFFN: Module {
        // Additional wrapper with explicit named components
        // (mlx-vlm uses NamedSequential as a generic container sequence with names)
        class ConvWithNorm: Module {
            @ModuleInfo var conv: Conv2d
            @ModuleInfo var bn: BatchNorm

            public init(inChannels: Int, outChannels: Int) {
                self.conv = Conv2d(
                    inputChannels: inChannels,
                    outputChannels: outChannels,
                    kernelSize: IntOrPair(7),
                    padding: IntOrPair(3),
                    groups: inChannels,
                    bias: false
                )
                self.bn = BatchNorm(featureCount: outChannels, trackRunningStats: true)
            }

            public func callAsFunction(_ x: MLXArray) -> MLXArray {
                bn(conv(x))
            }
        }

        let conv: ConvWithNorm
        @ModuleInfo var fc1: Conv2d
        let act: UnaryLayer
        @ModuleInfo var fc2: Conv2d

        public init(
            inChannels: Int, hiddenChannels: Int? = nil, outChannels: Int? = nil,
            activation: UnaryLayer = GELU()
        ) {
            let outChannels = outChannels ?? inChannels
            let hiddenChannels = hiddenChannels ?? inChannels

            self.conv = ConvWithNorm(inChannels: inChannels, outChannels: outChannels)
            self._fc1.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: hiddenChannels,
                kernelSize: IntOrPair(1)
            )
            self.act = activation
            self._fc2.wrappedValue = Conv2d(
                inputChannels: hiddenChannels,
                outputChannels: outChannels,
                kernelSize: IntOrPair(1)
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = conv(x)
            x = fc1(x)
            x = act(x)
            x = fc2(x)
            return x
        }
    }

    /// LayerNorm only for Channel Dimension
    /// Input: tensor in shape [B, H, W, C]
    fileprivate class LayerNormChannel: Module {
        let eps: Float
        let weight: MLXArray
        let bias: MLXArray

        public init(numFeatures: Int, eps: Float = 1e-05) {
            self.eps = eps
            self.weight = MLXArray.ones([numFeatures])
            self.bias = MLXArray.zeros([numFeatures])
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let u = x.mean(axis: -1, keepDims: true)
            let s = pow(x - u, 2).mean(axis: -1, keepDims: true)
            var result = (x - u) / sqrt(s + eps)
            result = weight * result + bias
            return result
        }
    }

    /// Squeeze and Excite module
    /// See `Squeeze-and-Excitation Networks` - https://arxiv.org/pdf/1709.01507.pdf
    fileprivate class SEBlock: Module, UnaryLayer {
        @ModuleInfo var reduce: Conv2d
        @ModuleInfo var expand: Conv2d

        public init(inChannels: Int, reductionRatio: Float = 0.0625) {
            self._reduce.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: Int(Float(inChannels) * reductionRatio),
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                bias: true
            )
            self._expand.wrappedValue = Conv2d(
                inputChannels: Int(Float(inChannels) * reductionRatio),
                outputChannels: inChannels,
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                bias: true
            )
        }

        public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
            let (b, h, w, c) = (inputs.dim(0), inputs.dim(1), inputs.dim(2), inputs.dim(3))

            var x = AvgPool2d(kernelSize: IntOrPair((h, w)), stride: IntOrPair((h, w)))(inputs)
            x = reduce(x)
            x = relu(x)
            x = expand(x)
            x = sigmoid(x)
            x = x.reshaped(b, 1, 1, c)
            return inputs * x
        }
    }

    /// MobileOne building block
    /// This implementation only uses the inference time CNN architecture and uses FastViTHD conventions
    fileprivate class MobileOneBlock: Module, UnaryLayer {
        let groups: Int
        let stride: Int
        let padding: Int
        let dilation: Int
        let kernelSize: Int
        let inChannels: Int
        let outChannels: Int

        let se: UnaryLayer
        let activation: GELU
        @ModuleInfo(key: "reparam_conv") var reparamConv: Conv2d

        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Int,
            stride: Int = 1,
            padding: Int = 0,
            dilation: Int = 1,
            groups: Int = 1,
            useSE: Bool = false
        ) {
            self.groups = groups
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.kernelSize = kernelSize
            self.inChannels = inChannels
            self.outChannels = outChannels

            if useSE {
                self.se = SEBlock(inChannels: outChannels)
            } else {
                self.se = Identity()
            }

            self.activation = GELU()
            self._reparamConv.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),
                groups: groups,
                bias: true
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            activation(se(reparamConv(x)))
        }
    }

    /// Building Block of RepLKNet
    /// This class defines overparameterized large kernel conv block
    /// introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_
    /// Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    fileprivate class ReparamLargeKernelConv: Module, UnaryLayer {
        let activation: GELU
        @ModuleInfo(key: "lkb_reparam") var lkbReparam: Conv2d

        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Int,
            stride: Int,
            groups: Int
        ) {
            self.activation = GELU()
            self._lkbReparam.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(kernelSize / 2),
                dilation: IntOrPair(1),
                groups: groups,
                bias: true
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            activation(lkbReparam(x))
        }
    }

    /// Convolutional patch embedding layer
    fileprivate class PatchEmbed: Module, UnaryLayer {
        let proj: [UnaryLayer]

        public init(
            patchSize: Int,
            stride: Int,
            inChannels: Int,
            embedDim: Int
        ) {
            self.proj = [
                ReparamLargeKernelConv(
                    inChannels: inChannels,
                    outChannels: embedDim,
                    kernelSize: patchSize,
                    stride: stride,
                    groups: inChannels
                ),
                MobileOneBlock(
                    inChannels: embedDim,
                    outChannels: embedDim,
                    kernelSize: 1,
                    stride: 1,
                    padding: 0,
                    groups: 1,
                    useSE: false
                ),
            ]
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var result = x
            for layer in proj {
                result = layer(result)
            }
            return result
        }
    }

    /// Implementation of conditional positional encoding
    /// For more details refer to paper:
    /// `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    fileprivate class RepCPE: Module, UnaryLayer {
        @ModuleInfo(key: "reparam_conv") var reparamConv: Conv2d

        public init(inChannels: Int, embedDim: Int = 768, spatialShape: (Int, Int) = (7, 7)) {
            self._reparamConv.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: embedDim,
                kernelSize: IntOrPair(spatialShape),
                stride: IntOrPair(1),
                padding: IntOrPair(spatialShape.0 / 2),
                groups: embedDim,
                bias: true
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            reparamConv(x)
        }
    }

    /// Reparameterizable token mixer
    /// For more details, please refer to Apple's paper:
    /// `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    fileprivate class RepMixer: Module {
        @ModuleInfo(key: "reparam_conv") var reparamConv: Conv2d

        public init(dim: Int, kernelSize: Int = 3) {
            self._reparamConv.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: dim,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(1),
                padding: IntOrPair(kernelSize / 2),
                groups: dim,
                bias: true
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            reparamConv(x)
        }
    }

    /// Implementation of Metaformer block with RepMixer as token mixer
    /// For more details on Metaformer structure, please refer to:
    /// `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    fileprivate class RepMixerBlock: Module, UnaryLayer {
        @ModuleInfo(key: "token_mixer") var tokenMixer: RepMixer
        @ModuleInfo(key: "convffn") var convFFN: ConvFFN
        @ModuleInfo(key: "layer_scale") var layerScale: MLXArray

        public init(dim: Int, kernelSize: Int = 3, mlpRatio: Float = 4) {
            precondition(mlpRatio > 0, "MLP ratio should be greater than 0, found: \(mlpRatio)")

            self._tokenMixer.wrappedValue = RepMixer(dim: dim, kernelSize: kernelSize)
            let mlpHiddenDim = Int(Float(dim) * mlpRatio)
            self._convFFN.wrappedValue = ConvFFN(
                inChannels: dim,
                hiddenChannels: mlpHiddenDim
            )
            self._layerScale.wrappedValue = MLXArray.ones([1, 1, dim])
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = tokenMixer(x)
            x = x + layerScale * convFFN(x)
            return x
        }
    }

    /// Implementation of metaformer block with MHSA as token mixer
    /// For more details on Metaformer structure, please refer to:
    /// `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    fileprivate class AttentionBlock: Module, UnaryLayer {
        @ModuleInfo(key: "norm") var norm: LayerNormChannel
        @ModuleInfo(key: "token_mixer") var tokenMixer: MHSA
        @ModuleInfo(key: "convffn") var convFFN: ConvFFN
        let layerScale1: MLXArray
        let layerScale2: MLXArray

        public init(dim: Int, mlpRatio: Float = 4.0) {
            precondition(mlpRatio > 0, "MLP ratio should be greater than 0, found: \(mlpRatio)")

            _norm.wrappedValue = LayerNormChannel(numFeatures: dim)
            self._tokenMixer.wrappedValue = MHSA(dim: dim)

            let mlpHiddenDim = Int(Float(dim) * mlpRatio)
            self._convFFN.wrappedValue = ConvFFN(
                inChannels: dim,
                hiddenChannels: mlpHiddenDim
            )

            self.layerScale1 = MLXArray.ones([1, 1, dim])
            self.layerScale2 = MLXArray.ones([1, 1, dim])
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = x + layerScale1 * tokenMixer(norm(x))
            x = x + layerScale2 * convFFN(x)
            return x
        }
    }

    fileprivate static func basicBlocks(
        dim: Int,
        blockIndex: Int,
        numBlocks: [Int],
        tokenMixerType: String,
        kernelSize: Int = 3,
        mlpRatio: Float = 4.0
    ) -> UnaryLayer {
        var blocks = [UnaryLayer]()
        for _ in 0 ..< numBlocks[blockIndex] {
            if tokenMixerType == "repmixer" {
                blocks.append(
                    RepMixerBlock(
                        dim: dim,
                        kernelSize: kernelSize,
                        mlpRatio: mlpRatio
                    )
                )
            } else if tokenMixerType == "attention" {
                blocks.append(
                    AttentionBlock(
                        dim: dim,
                        mlpRatio: mlpRatio
                    )
                )
            } else {
                fatalError("Token mixer type: \(tokenMixerType) not supported")
            }
        }
        return Sequential(layers: blocks)
    }

    fileprivate static func buildFastViTNetwork(config: FastVLMConfiguration.VisionConfiguration)
        -> [UnaryLayer]
    {
        var network = [UnaryLayer]()
        for i in 0 ..< config.layers.count {
            if let spatialShape = config.posEmbedShapes[i] {
                let positionEmbeddings = RepCPE(
                    inChannels: config.embedDimensions[i],
                    embedDim: config.embedDimensions[i],
                    spatialShape: (spatialShape[0], spatialShape[1])
                )
                network.append(positionEmbeddings)
            }

            let stage = basicBlocks(
                dim: config.embedDimensions[i],
                blockIndex: i,
                numBlocks: config.layers,
                tokenMixerType: config.tokenMixers[i],
                kernelSize: config.repMixerKernelSize,
                mlpRatio: Float(config.mlpRatios[i])
            )
            network.append(stage)

            if i >= config.layers.count - 1 {
                break
            }

            // Patch merging/downsampling between stages
            if config.downSamples[i] || config.embedDimensions[i] != config.embedDimensions[i + 1] {
                network.append(
                    PatchEmbed(
                        patchSize: config.downPatchSize,
                        stride: config.downStride,
                        inChannels: config.embedDimensions[i],
                        embedDim: config.embedDimensions[i + 1]
                    )
                )
            }
        }

        return network
    }

    /// Convolutional stem
    fileprivate class ConvolutionalStem: Module {
        let blocks: [MobileOneBlock]

        public init(config: FastVLMConfiguration.VisionConfiguration) {
            let inChannels = 3
            let outChannels = config.embedDimensions[0]

            self.blocks = [
                MobileOneBlock(
                    inChannels: inChannels,
                    outChannels: outChannels,
                    kernelSize: 3,
                    stride: 2,
                    padding: 1,
                    groups: 1
                ),
                MobileOneBlock(
                    inChannels: outChannels,
                    outChannels: outChannels,
                    kernelSize: 3,
                    stride: 2,
                    padding: 1,
                    groups: outChannels
                ),
                MobileOneBlock(
                    inChannels: outChannels,
                    outChannels: outChannels,
                    kernelSize: 1,
                    stride: 1,
                    padding: 0,
                    groups: 1
                ),
            ]
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var result = x
            for block in blocks {
                result = block(result)
            }
            return result
        }
    }

    /// This class implements global pooling with linear projection
    fileprivate class GlobalPool2D: Module, UnaryLayer {
        let proj: MLXArray

        public init(inDim: Int, outDim: Int) {
            self.proj = MLXArray.zeros([inDim, outDim])
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            precondition(
                x.ndim == 4,
                "Input should be 4-dimensional (Batch x in_height x in_width x in_dim). Got: \(x.shape)"
            )

            // [batch, in_height, in_width, in_dim] --> [batch, in_dim]
            var result = x.mean(axes: [1, 2])
            // [batch, in_dim] x [in_dim, out_dim] --> [batch, out_dim]
            result = matmul(result, proj)
            return result
        }
    }

    /// FastViTHD Model
    /// Based on https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/multimodal_encoder/mobileclip/mci.py
    /// Hardcoded, for now, for:
    /// - FastViTHD variant
    /// - Use inference_mode (i.e., modules contain the convolutional reparameterized versions of the architecture)
    fileprivate class FastViTHDModel: Module {
        let config: FastVLMConfiguration.VisionConfiguration
        @ModuleInfo(key: "patch_embed") var patchEmbed: ConvolutionalStem
        let network: [UnaryLayer]
        @ModuleInfo(key: "conv_exp") var convExp: MobileOneBlock
        let head: UnaryLayer

        public init(config: FastVLMConfiguration.VisionConfiguration) {
            self.config = config

            self._patchEmbed.wrappedValue = ConvolutionalStem(config: config)
            self.network = buildFastViTNetwork(config: config)
            self._convExp.wrappedValue = MobileOneBlock(
                inChannels: config.embedDimensions.last!,
                outChannels: Int(Float(config.embedDimensions.last!) * config.classHeadRatio),
                kernelSize: 3,
                stride: 1,
                padding: 1,
                groups: config.embedDimensions.last!,
                useSE: true
            )
            // Replaced head
            // https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/multimodal_encoder/mobileclip/__init__.py#L49
            let inDim = Int(Float(config.embedDimensions.last!) * config.classHeadRatio)
            head = GlobalPool2D(inDim: inDim, outDim: config.projectionDim)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?
        ) {
            var x = patchEmbed(x)

            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for layer in network {
                x = layer(x)
                if outputHiddenStates {
                    encoderStates?.append(x)
                }
            }

            x = convExp(x)
            let clsOut = head(x)

            return (clsOut, x, encoderStates)
        }
    }

    /// Vision Model wrapper
    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: FastViTHDModel

        public init(_ config: FastVLMConfiguration.VisionConfiguration) {
            self._visionModel.wrappedValue = FastViTHDModel(config: config)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights: [String: MLXArray] = [:]
            for (k, v) in weights {
                var key = k
                if key.contains("layer_scale_") {
                    key = key.replacingOccurrences(of: "layer_scale_", with: "layerScale")
                }
                if key.contains("vision_model.network") {
                    // vision_model.network.0.1 -> vision_model.network.0.layers.1
                    // (i.e., vision_model.network.0.proj not transformed)
                    let regex = #/(.+)\.vision_model.network.(\d+)\.(\d+)\.(.+)/#
                    if let match = key.firstMatch(of: regex) {
                        key =
                            "\(match.1).vision_model.network.\(match.2).layers.\(match.3).\(match.4)"
                    }
                }
                sanitizedWeights[key] = v
            }
            return sanitizedWeights
        }
    }
}

// MARK: - Processor

public struct FastVLMProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let width: Int
        public let height: Int

        enum CodingKeys: String, CodingKey {
            case width
            case height
        }

        var cgSize: CGSize { CGSize(width: width, height: height) }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let cropSize: Size

    // NOTE: Hardcoded config values and assumptions
    // - crop_size matches size.shortest_edge
    // - bicubic interpolation
    // - scale by multiplying by 1/255

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case cropSize = "crop_size"
    }
}

public class FastVLMProcessor: UserInputProcessor {
    private let config: FastVLMProcessorConfiguration
    private let tokenizer: any Tokenizer

    private let imageToken = "<image>"
    private let imageTokenIndex = -200

    public init(
        _ config: FastVLMProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: MLXLMCommon.UserInput) async throws -> MLXLMCommon.LMInput {
        let messages = FastVLMMessageGenerator().generate(from: input)

        if input.images.isEmpty {
            // No image scenario
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        }

        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        // Unfortunately we don't have a "render" option in Tokenizers yet, so decoding
        let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
        let decoded = try tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)

        // Find <image> and replace with token id -200
        let pieces = decoded.split(separator: imageToken)
        let tokens = Array(
            pieces.map { tokenizer.encode(text: String($0)) }.joined(separator: [-200]))

        let image = try input.images[0]
            .asCIImage()
            .toSRGB()
            .paddingToSquare()
            .resampled(to: config.cropSize.cgSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)

        let pixels = image.asMLXArray()

        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: pixels)
        )
    }
}

// MARK: - Multimodal Projector

private class FastVLMMultiModalProjector: Sequential {
    init(_ config: FastVLMConfiguration) {
        let hiddenSize = config.textConfiguration.hiddenSize
        let mlpGeluRegex = #/^mlp(\d+)x_gelu$/#
        guard
            let match = config.baseConfiguration.multimodalProjectorType.firstMatch(
                of: mlpGeluRegex)
        else {
            // Fall back to Linear if no match
            super.init(layers: [
                Linear(config.baseConfiguration.multimodalProjectorHiddenSize, hiddenSize)
            ])
            return
        }

        let mlpDepth = Int(match.1) ?? 2
        super.init {
            Linear(config.baseConfiguration.multimodalProjectorHiddenSize, hiddenSize)
            for _ in 1 ..< mlpDepth {
                GELU()
                Linear(hiddenSize, hiddenSize)
            }
        }
    }
}

// MARK: - Model

public class FastVLM: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "mm_projector") private var multimodalProjector: FastVLMMultiModalProjector
    public let config: FastVLMConfiguration

    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: FastVLMConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
        self._multimodalProjector.wrappedValue = FastVLMMultiModalProjector(config)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func getInputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, mask: MLXArray?)
        -> MLXArray
    {
        guard let pixelValues = pixelValues else {
            return languageModel.model.embedTokens(inputIds)
        }

        let (_, imageFeatures, _) = visionModel(pixelValues.transposed(0, 2, 3, 1))
        let (B, H, W, C) = (
            imageFeatures.shape[0], imageFeatures.shape[1], imageFeatures.shape[2],
            imageFeatures.shape[3]
        )
        let mmInputs = multimodalProjector(imageFeatures.reshaped(B, H * W, C))
        let finalEmbeddings = prepareInputsForMultimodal(
            imageFeatures: mmInputs, inputIds: inputIds, mask: mask)
        return finalEmbeddings
    }

    // This method assumes bs == 1, and one single image
    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray, inputIds ids: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let inputIds: MLXArray
        if let mask = mask {
            // Remove padding
            let start: Int = mask[0].argMax().item()
            let end = start + mask[0].sum().item()
            inputIds = ids[0][start ..< end]
        } else {
            inputIds = ids
        }

        let inputIdsArray = inputIds.asArray(Int.self)
        let imageTokenIndex = inputIdsArray.index(of: config.baseConfiguration.imageTokenIndex) ?? 0
        // Embed tokens before and after and then split to insert the image
        let tokens = inputIdsArray.split(separator: config.baseConfiguration.imageTokenIndex)
            .joined()
        let tokenEmbeddings = languageModel.model.embedTokens(MLXArray(tokens))
        let splitTokenEmbeddings = tokenEmbeddings.split(indices: [imageTokenIndex])

        // Concatenate - once again this is easy because we assume bs==1 and a single image
        let embeddings = concatenated(
            [splitTokenEmbeddings[0], imageFeatures[0], splitTokenEmbeddings[1]], axis: 0)

        // TODO: trim if we went over model_max_length
        return embeddings.expandedDimensions(axis: 0)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let embeddings = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            mask: input.text.mask
        )
        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let out = languageModel(inputs, cache: cache).logits
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Not sure we need to replicate the full Python logic since the weights were transformed on conversion

        var sanitizedWeights: [String: MLXArray] = [:]
        for (k, v) in weights {
            var key = k
            if key.contains("mm_projector") {
                key = key.replacingOccurrences(of: "mm_projector", with: "mm_projector.layers")
            }
            sanitizedWeights[key] = v
        }
        return visionModel.sanitize(weights: sanitizedWeights)
    }
}

// MARK: - Message Generator

/// This message generator adheres to the following format:
/// - Image precedes text content
/// - Empty system messages are removed - the chat template applies a default one in this case
public struct FastVLMMessageGenerator: MessageGenerator {
    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": []
                + message.images.map { _ in
                    ["type": "image"]
                }
                + [["type": "text", "text": message.content]],
        ]
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        // Remove system role if empty, because the template adds a default one
        messages
            .filter { $0.role != .system || ($0.role == .system && !$0.content.isEmpty) }
            .map { generate(message: $0) }
    }
}
