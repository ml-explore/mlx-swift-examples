//
//  LoRA+Container.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN

/// A container for managing LoRA or DoRA adapters and applying them to a language model.
///
/// This struct conforms to `ModelAdapter` and can dynamically inject, remove, or fuse adapters into a model at runtime.
public struct LoRAContainer: ModelAdapter {

    /// Configuration for how LoRA or DoRA should be applied.
    ///
    /// Note: It's compatible with `adapter_config.json` file created during training using MLX libraries.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "fine_tune_type": "lora",
    ///   "num_layers": 28,
    ///   "lora_parameters": {
    ///     "rank": 16,
    ///     "scale": 20.0
    ///   }
    /// }
    /// ```
    public struct Configuration: Codable {

        public enum FineTuneType: String, Codable {
            case lora
            case dora
        }

        public struct LoRAParameters: Codable {
            public let rank: Int
            public let scale: Float
        }

        public let numLayers: Int
        public let fineTuneType: FineTuneType
        public let loraParameters: LoRAParameters
    }

    /// The configuration used to construct this adapter container.
    public let configuration: Configuration
    /// The parameter values for the adapter modules.
    public let parameters: ModuleParameters

    public init(
        configuration: Configuration,
        parameters: ModuleParameters
    ) {
        self.configuration = configuration
        self.parameters = parameters
    }

    /// Applies adapter modules (LoRA or DoRA) to the given model.
    ///
    /// This method replaces target linear layers in the model with corresponding
    /// adapter layers based on the configuration. It also loads adapter-specific
    /// weights into the model.
    public func load(into model: LanguageModel) throws {
        try replaceLayers(model: model) { (linear: Linear) in
            createReplacementLayer(linear)
        }

        try model.update(
            parameters: parameters,
            verify: .noUnusedKeys
        )
    }

    /// Removes adapter layers (LoRA or DoRA) and restores the model to its original form.
    ///
    /// This method reverts each adapted layer to its original linear layer, if possible.
    public func unload(from model: LanguageModel) {
        try? replaceLayers(model: model) { (lora: LoRALayer) in
            lora.reverted()
        }
    }

    /// Permanently fuses the adapter weights into the model's base layers.
    ///
    /// After fusion, adapter weights become part of the model’s original parameters,
    /// and adapter layers are no longer needed.
    public func fuse(with model: LanguageModel) throws {
        try replaceLayers(model: model) { (linear: Linear) in
            if let lora = linear as? LoRALayer {
                lora.fused()
            } else {
                createReplacementLayer(linear).fused()
            }
        }
    }

    /// Creates an adapter replacement layer for a given linear layer based on the configuration.
    private func createReplacementLayer(_ linear: Linear) -> LoRALayer {
        switch configuration.fineTuneType {
        case .lora:
            LoRALinear.from(
                linear: linear,
                rank: configuration.loraParameters.rank,
                scale: configuration.loraParameters.scale
            )
        case .dora:
            DoRALinear.from(
                linear: linear,
                rank: configuration.loraParameters.rank,
                scale: configuration.loraParameters.scale
            )
        }
    }

    /// Traverses the model and replaces its layers using a transformation closure.
    private func replaceLayers<T>(model: LanguageModel, transforming transform: (T) -> Module?)
        throws
    {
        guard let lora = model as? LoRAModel else {
            throw ModelAdapterError.incompatibleModelType
        }

        let layers = lora.loraLinearLayers(configuration.numLayers)
        for (layer, keys) in layers {
            var update = ModuleChildren()
            let children = layer.children()

            for key in keys {
                if let item = children[key], case .value(let child) = item {
                    if let child = child as? T, let transformed = transform(child) {
                        update[key] = .value(transformed)
                    }
                }
            }

            if !update.isEmpty {
                layer.update(modules: update)
            }
        }
    }
}
