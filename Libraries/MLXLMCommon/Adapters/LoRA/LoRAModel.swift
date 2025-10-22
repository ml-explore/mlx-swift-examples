//
//  LoRAModel.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 03.06.2025.
//

import Foundation
import MLX
import MLXNN

public protocol LoRAModel {

    /// Return the layers to apply LoRA adapters to.
    ///
    /// Typically, this includes all transformer layers.
    /// Must be defined explicitly since we can't unify it across all models.
    var loraLayers: [Module] { get }

    /// Default layer keys to apply LoRA adapters to.
    ///
    /// Used when not specified in `adapter_config.json`.
    /// Otherwise, keys from the config are applied.
    var loraDefaultKeys: [String] { get }
}

extension LoRAModel {

    /// By default we apply LoRA to all Linear layers.
    /// This is aligned with `mlx-lm` Python logic.
    public var loraDefaultKeys: [String] {
        let namedModules = loraLayers.flatMap { $0.namedModules() }
        let linearKeys = namedModules.compactMap { key, module in
            if module is Linear {
                return key
            } else {
                return nil
            }
        }
        let unique = Set(linearKeys)
        return Array(unique)
    }
}

/// A protocol representing a module that includes a LoRA adapter and can be converted
/// back to its original, unadapted form.
public protocol LoRALayer: Module {

    /// Returns a version of the module with the LoRA adapter permanently fused in.
    func fused() -> Module

    /// Returns the original module, without the LoRA adapter applied.
    func reverted() -> Module
}

/// Default implementation of `reverted()` for `Linear` layers, including support for quantized layers.
extension LoRALayer where Self: Linear {
    public func reverted() -> Module {
        if let quantized = self as? QuantizedLinear {
            return QuantizedLinear(
                weight: quantized.weight, bias: quantized.bias,
                scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits
            )
        } else {
            return Linear(weight: weight, bias: bias)
        }
    }
}

/// Extension for `QuantizedLinear` to provide helper properties.
extension QuantizedLinear {

    /// Computes the dequantized weight matrix using the stored quantization parameters.
    var dequantizedWeight: MLXArray {
        dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}
