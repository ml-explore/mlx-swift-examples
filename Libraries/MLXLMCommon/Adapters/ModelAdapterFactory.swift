//
//  ModelAdapterFactory.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 03.06.2025.
//

import Foundation
import MLX
import MLXNN
import Hub

/// A factory responsible for loading and creating model adapters from hub configurations.
public final class ModelAdapterFactory {
    
    /// Shared instance of the adapter factory.
    public static let shared = ModelAdapterFactory()
    
    /// Loads a model adapter from the hub using the provided model configuration.
    ///
    /// This method fetches the adapter configuration and weights, decodes the appropriate
    /// fine-tuning format, and initializes a `ModelAdapter` accordingly.
    ///
    /// Supports fine-tuning types like `LoRA` and `DoRA`.
    public func load(
        hub: HubApi = HubApi(),
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelAdapter {
        
        struct BaseConfiguration: Decodable {
            
            enum FineTuneType: String, Decodable {
                case lora
                case dora
            }
            
            let fineTuneType: FineTuneType
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        
        let snapshotURL = try await downloadModel(hub: hub, configuration: configuration, progressHandler: progressHandler)
        let configurationURL = snapshotURL.appending(path: "adapter_config").appendingPathExtension("json")
        let configurationData = try Data(contentsOf: configurationURL)
        let configuration = try decoder.decode(BaseConfiguration.self, from: configurationData)
        
        let weightsURL = snapshotURL.appending(path: "adapters").appendingPathExtension("safetensors")
        let weights = try MLX.loadArrays(url: weightsURL)
        let parameters = ModuleParameters.unflattened(weights)
        
        switch configuration.fineTuneType {
        case .lora, .dora:
            return LoRAContainer(
                configuration: try decoder.decode(LoRAContainer.Configuration.self, from: configurationData),
                parameters: parameters
            )
        }
    }
}
