// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/config.py

/// Configuration for ``Autoencoder``
struct AutoencoderConfiguration: Codable {

    public var inputChannels = 3
    public var outputChannels = 3
    public var latentChannelsOut: Int { latentChannelsIn * 2 }
    public var latentChannelsIn = 4
    public var blockOutChannels = [128, 256, 512, 512]
    public var layersPerBlock = 2
    public var normNumGroups = 32
    public var scalingFactor: Float = 0.18215

    enum CodingKeys: String, CodingKey {
        case inputChannels = "in_channels"
        case outputChannels = "out_channels"
        case latentChannelsIn = "latent_channels"
        case blockOutChannels = "block_out_channels"
        case layersPerBlock = "layers_per_block"
        case normNumGroups = "norm_num_groups"
        case scalingFactor = "scaling_factor"
    }

    public init(from decoder: any Decoder) throws {
        let container: KeyedDecodingContainer<AutoencoderConfiguration.CodingKeys> =
            try decoder.container(keyedBy: AutoencoderConfiguration.CodingKeys.self)

        // load_autoencoder()

        self.scalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .scalingFactor) ?? 0.18215

        self.inputChannels = try container.decode(Int.self, forKey: .inputChannels)
        self.outputChannels = try container.decode(Int.self, forKey: .outputChannels)
        self.latentChannelsIn = try container.decode(Int.self, forKey: .latentChannelsIn)
        self.blockOutChannels = try container.decode([Int].self, forKey: .blockOutChannels)
        self.layersPerBlock = try container.decode(Int.self, forKey: .layersPerBlock)
        self.normNumGroups = try container.decode(Int.self, forKey: .normNumGroups)
    }

    public func encode(to encoder: any Encoder) throws {
        var container: KeyedEncodingContainer<AutoencoderConfiguration.CodingKeys> =
            encoder.container(keyedBy: AutoencoderConfiguration.CodingKeys.self)

        try container.encode(self.inputChannels, forKey: .inputChannels)
        try container.encode(self.outputChannels, forKey: .outputChannels)
        try container.encode(self.latentChannelsIn, forKey: .latentChannelsIn)
        try container.encode(self.blockOutChannels, forKey: .blockOutChannels)
        try container.encode(self.layersPerBlock, forKey: .layersPerBlock)
        try container.encode(self.normNumGroups, forKey: .normNumGroups)
        try container.encode(self.scalingFactor, forKey: .scalingFactor)
    }
}

/// Configuration for ``CLIPTextModel``
struct CLIPTextModelConfiguration: Codable {

    public enum ClipActivation: String, Codable {
        case fast = "quick_gelu"
        case gelu = "gelu"

        var activation: (MLXArray) -> MLXArray {
            switch self {
            case .fast: MLXNN.geluFastApproximate
            case .gelu: MLXNN.gelu
            }
        }
    }

    public var numLayers = 23
    public var modelDimensions = 1024
    public var numHeads = 16
    public var maxLength = 77
    public var vocabularySize = 49408
    public var projectionDimensions: Int? = nil
    public var hiddenActivation: ClipActivation = .fast

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_hidden_layers"
        case modelDimensions = "hidden_size"
        case numHeads = "num_attention_heads"
        case maxLength = "max_position_embeddings"
        case vocabularySize = "vocab_size"
        case projectionDimensions = "projection_dim"
        case hiddenActivation = "hidden_act"
        case architectures = "architectures"
    }

    public init(from decoder: any Decoder) throws {
        let container: KeyedDecodingContainer<CLIPTextModelConfiguration.CodingKeys> =
            try decoder.container(keyedBy: CLIPTextModelConfiguration.CodingKeys.self)

        // see load_text_encoder

        let architectures = try container.decode([String].self, forKey: .architectures)
        let withProjection = architectures[0].contains("WithProjection")

        self.projectionDimensions =
            withProjection
            ? try container.decodeIfPresent(Int.self, forKey: .projectionDimensions) : nil
        self.hiddenActivation =
            try container.decodeIfPresent(
                CLIPTextModelConfiguration.ClipActivation.self, forKey: .hiddenActivation) ?? .fast

        self.numLayers = try container.decode(Int.self, forKey: .numLayers)
        self.modelDimensions = try container.decode(Int.self, forKey: .modelDimensions)
        self.numHeads = try container.decode(Int.self, forKey: .numHeads)
        self.maxLength = try container.decode(Int.self, forKey: .maxLength)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
    }

    public func encode(to encoder: any Encoder) throws {
        var container: KeyedEncodingContainer<CLIPTextModelConfiguration.CodingKeys> =
            encoder.container(keyedBy: CLIPTextModelConfiguration.CodingKeys.self)

        if projectionDimensions != nil {
            try container.encode(["WithProjection"], forKey: .architectures)
        } else {
            try container.encode(["Other"], forKey: .architectures)
        }

        try container.encode(self.numLayers, forKey: .numLayers)
        try container.encode(self.modelDimensions, forKey: .modelDimensions)
        try container.encode(self.numHeads, forKey: .numHeads)
        try container.encode(self.maxLength, forKey: .maxLength)
        try container.encode(self.vocabularySize, forKey: .vocabularySize)
        try container.encodeIfPresent(self.projectionDimensions, forKey: .projectionDimensions)
        try container.encode(self.hiddenActivation, forKey: .hiddenActivation)
    }
}

/// Configuration for ``UNetModel``
struct UNetConfiguration: Codable {

    public var inputChannels = 4
    public var outputChannels = 4
    public var convolutionInKernel = 3
    public var convolutionOutKernel = 3
    public var blockOutChannels = [320, 640, 1280, 1280]
    public var layersPerBlock = [2, 2, 2, 2]
    public var midBlockLayers = 2
    public var transformerLayersPerBlock = [2, 2, 2, 2]
    public var numHeads = [5, 10, 20, 20]
    public var crossAttentionDimension = [1024, 1024, 1024, 1024]
    public var normNumGroups = 32
    public var downBlockTypes: [String] = []
    public var upBlockTypes: [String] = []
    public var additionEmbedType: String? = nil
    public var additionTimeEmbedDimension: Int? = nil
    public var projectionClassEmbeddingsInputDimension: Int? = nil

    enum CodingKeys: String, CodingKey {
        case inputChannels = "in_channels"
        case outputChannels = "out_channels"
        case convolutionInKernel = "conv_in_kernel"
        case convolutionOutKernel = "conv_out_kernel"
        case blockOutChannels = "block_out_channels"
        case layersPerBlock = "layers_per_block"
        case midBlockLayers = "mid_block_layers"
        case transformerLayersPerBlock = "transformer_layers_per_block"
        case numHeads = "attention_head_dim"
        case crossAttentionDimension = "cross_attention_dim"
        case normNumGroups = "norm_num_groups"
        case downBlockTypes = "down_block_types"
        case upBlockTypes = "up_block_types"
        case additionEmbedType = "addition_embed_type"
        case additionTimeEmbedDimension = "addition_time_embed_dim"
        case projectionClassEmbeddingsInputDimension = "projection_class_embeddings_input_dim"
    }

    public init() {
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<UNetConfiguration.CodingKeys> = try decoder.container(
            keyedBy: UNetConfiguration.CodingKeys.self)

        // customizations based on def load_unet(key: str = _DEFAULT_MODEL, float16: bool = False):
        //
        // Note: the encode() writes out the internal format (and this can load it back in)

        self.blockOutChannels = try container.decode([Int].self, forKey: .blockOutChannels)
        let nBlocks = blockOutChannels.count

        self.layersPerBlock =
            try (try? container.decode([Int].self, forKey: .layersPerBlock))
            ?? Array(repeating: container.decode(Int.self, forKey: .layersPerBlock), count: nBlocks)
        self.transformerLayersPerBlock =
            (try? container.decode([Int].self, forKey: .transformerLayersPerBlock)) ?? [1, 1, 1, 1]
        self.numHeads =
            try (try? container.decodeIfPresent([Int].self, forKey: .numHeads))
            ?? Array(repeating: container.decode(Int.self, forKey: .numHeads), count: nBlocks)
        self.crossAttentionDimension =
            try (try? container.decode([Int].self, forKey: .crossAttentionDimension))
            ?? Array(
                repeating: container.decode(Int.self, forKey: .crossAttentionDimension),
                count: nBlocks)
        self.upBlockTypes = try container.decode([String].self, forKey: .upBlockTypes).reversed()

        self.convolutionInKernel =
            try container.decodeIfPresent(Int.self, forKey: .convolutionInKernel) ?? 3
        self.convolutionOutKernel =
            try container.decodeIfPresent(Int.self, forKey: .convolutionOutKernel) ?? 3
        self.midBlockLayers = try container.decodeIfPresent(Int.self, forKey: .midBlockLayers) ?? 2

        self.inputChannels = try container.decode(Int.self, forKey: .inputChannels)
        self.outputChannels = try container.decode(Int.self, forKey: .outputChannels)
        self.normNumGroups = try container.decode(Int.self, forKey: .normNumGroups)
        self.downBlockTypes = try container.decode([String].self, forKey: .downBlockTypes)
        self.additionEmbedType = try container.decodeIfPresent(
            String.self, forKey: .additionEmbedType)
        self.additionTimeEmbedDimension = try container.decodeIfPresent(
            Int.self, forKey: .additionTimeEmbedDimension)
        self.projectionClassEmbeddingsInputDimension = try container.decodeIfPresent(
            Int.self, forKey: .projectionClassEmbeddingsInputDimension)
    }

    public func encode(to encoder: Encoder) throws {
        var container: KeyedEncodingContainer<UNetConfiguration.CodingKeys> = encoder.container(
            keyedBy: UNetConfiguration.CodingKeys.self)

        try container.encode(self.upBlockTypes.reversed(), forKey: .upBlockTypes)

        try container.encode(self.inputChannels, forKey: .inputChannels)
        try container.encode(self.outputChannels, forKey: .outputChannels)
        try container.encode(self.convolutionInKernel, forKey: .convolutionInKernel)
        try container.encode(self.convolutionOutKernel, forKey: .convolutionOutKernel)
        try container.encode(self.blockOutChannels, forKey: .blockOutChannels)
        try container.encode(self.layersPerBlock, forKey: .layersPerBlock)
        try container.encode(self.midBlockLayers, forKey: .midBlockLayers)
        try container.encode(self.transformerLayersPerBlock, forKey: .transformerLayersPerBlock)
        try container.encode(self.numHeads, forKey: .numHeads)
        try container.encode(self.crossAttentionDimension, forKey: .crossAttentionDimension)
        try container.encode(self.normNumGroups, forKey: .normNumGroups)
        try container.encode(self.downBlockTypes, forKey: .downBlockTypes)
        try container.encodeIfPresent(self.additionEmbedType, forKey: .additionEmbedType)
        try container.encodeIfPresent(
            self.additionTimeEmbedDimension, forKey: .additionTimeEmbedDimension)
        try container.encodeIfPresent(
            self.projectionClassEmbeddingsInputDimension,
            forKey: .projectionClassEmbeddingsInputDimension)
    }
}

/// Configuration for ``StableDiffusion``
public struct DiffusionConfiguration: Codable {

    public enum BetaSchedule: String, Codable {
        case linear = "linear"
        case scaledLinear = "scaled_linear"
    }

    public var betaSchedule = BetaSchedule.scaledLinear
    public var betaStart: Float = 0.00085
    public var betaEnd: Float = 0.012
    public var trainSteps = 3

    enum CodingKeys: String, CodingKey {
        case betaSchedule = "beta_schedule"
        case betaStart = "beta_start"
        case betaEnd = "beta_end"
        case trainSteps = "num_train_timesteps"
    }
}
