//
//  LoRAModel.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 03.06.2025.
//

import Foundation
import MLX
import MLXNN

/// Layers to apply LoRA adapters to.
///
/// This is the value returned by ``LoRAModel/loraLinearLayers()``.
public typealias LoRALinearLayers = [(Module, [String])]

public protocol LoRAModel {
    /// Return the layers and keys to apply LoRA adapters to.
    ///
    /// For example this might apply the adapters to the `q` an `v` projections in the
    /// Attention layers:
    ///
    /// ```swift
    /// model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    /// ```
    ///
    /// It is not required that a model implement this protocol to have LoRA adapters applied, but
    /// the command line driver example uses this to produce the ``LoRALinearLayers``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    func loraLinearLayers() -> LoRALinearLayers

    /// Return a suffix of the layers and keys to apply LoRA adapters to.
    ///
    /// See ``loraLinearLayers()``
    func loraLinearLayers(_ count: Int) -> LoRALinearLayers
}

extension LoRAModel {
    public func loraLinearLayers(_ count: Int) -> LoRALinearLayers {
        loraLinearLayers().suffix(count)
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
            bits: bits
        )
    }
}
