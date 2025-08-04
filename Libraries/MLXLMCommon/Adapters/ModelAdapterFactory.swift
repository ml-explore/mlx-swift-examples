//
//  ModelAdapterFactory.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 03.06.2025.
//

import Foundation
import Hub
import MLX
import MLXNN

/// Base configuration for any adapter.
///
/// This struct is parsed by `ModelAdapterFactory` to determine which adapter creator
/// to invoke from the registry. It expects an `adapter_config.json` file containing
/// a `fine_tune_type` field that specifies the adapter type as a string (e.g., "lora", "dora").
///
/// Note: This configuration does not consider adapter-specific parameters.
///
/// Example:
/// ```json
/// {
///   "fine_tune_type": "lora",     // Required
///   "additional_field": true      // Ignored here
/// }
/// ```
private struct ModelAdapterBaseConfiguration: Decodable {

    let fineTuneType: String

    enum CodingKeys: String, CodingKey {
        case fineTuneType = "fine_tune_type"
    }
}

/// A factory responsible for loading and creating model adapters from hub configurations.
public final class ModelAdapterFactory {

    /// Shared instance of the adapter factory.
    public static let shared = ModelAdapterFactory(
        registry: ModelAdapterTypeRegistry(creators: [
            "lora": LoRAContainer.from(directory:),
            "dora": LoRAContainer.from(directory:),
        ])
    )

    /// Registry of adapter type creators.
    ///
    /// You can register custom adapter types as follows:
    /// ```swift
    /// let registry = ModelAdapterFactory.shared.registry
    /// let adapterType = "my-adapter"
    /// let adapterCreator = MyAdapterContainer.from(directory:)
    /// registry.registerAdapterType(adapterType, creator: adapterCreator)
    /// ```
    ///
    /// This allows the factory to load your custom adapter from automatically
    /// when the matching type is found in a configuration file.
    public let registry: ModelAdapterTypeRegistry

    public init(registry: ModelAdapterTypeRegistry) {
        self.registry = registry
    }

    /// Loads a model adapter from the hub using the provided model configuration.
    ///
    /// This method fetches the adapter configuration and weights, decodes the appropriate
    /// fine-tuning format, and initializes a `ModelAdapter` accordingly.
    public func load(
        hub: HubApi = HubApi(),
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelAdapter {
        let adapterDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler
        )

        let configurationURL = adapterDirectory.appending(component: "adapter_config.json")
        let configurationData = try Data(contentsOf: configurationURL)
        let configuration = try JSONDecoder()
            .decode(ModelAdapterBaseConfiguration.self, from: configurationData)

        return try registry.createAdapter(
            directory: adapterDirectory,
            adapterType: configuration.fineTuneType
        )
    }
}
