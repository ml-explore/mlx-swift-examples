// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Tokenizers

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

/// Protocol for LoRA implementations that provides a method for converting back to a `Linear`
/// (or subtype).
///
/// This is normally called via ``LoRATrain/fuse(model:layers:deQuantize:)``
public protocol LoRAConvertToLinear {
    func toLinear(deQuantize: Bool) -> Linear
}

/// Implementation of LoRA `Linear` replacement layer.
///
/// This layer implements the LoRA capabilities for `Linear` layers, specifically:
///
/// - converting `Linear` or `QuantizedLinear` layers to ``LoRALinear`` / ``QLoRALinear``
/// - converting ``LoRALinear`` back to `Linear` or `QuantizedLinear` (``LoRAConvertToLinear``)
/// - implementing the LoRA evaluation
///
/// ``QLoRALinear`` is the equivalent class for `QuantizedLinear`.
///
/// This is not typically used directly -- ``LoRATrain/convert(model:layers:)`` is used to
/// add the adapter layers to a given model.
///
/// ### See Also
/// - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
/// - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
/// - ``QLoRALinear``
/// - ``LoRATrain/convert(model:layers:)``
/// - ``LoRATrain/fuse(model:layers:deQuantize:)``
public class LoRALinear: Linear, LoRAConvertToLinear {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: Linear
    ) {
        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    /// - ``QLoRALinear/from(linear:rank:)``
    public static func from(linear: Linear, rank: Int = 8) -> Linear {
        if let linear = linear as? QuantizedLinear {
            return QLoRALinear.from(linear: linear, rank: rank)
        }
        let (outputDimensions, inputDimensions) = linear.shape
        return LoRALinear(inputDimensions, outputDimensions, rank: rank, linear: linear)
    }

    /// Convert back into a fused `Linear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:layers:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:layers:deQuantize:)``
    /// - ``LoRAConvertToLinear``
    /// - ``QLoRALinear/toLinear(deQuantize:)``
    public func toLinear(deQuantize: Bool = false) -> Linear {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return Linear(weight: weight + matmul(loraB, loraA), bias: bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}

/// Implementation of LoRA `QuantizedLinear` replacement layer.
///
/// See ``LoRALinear`` (equivalent class for `Linear` layers) for more information.
public class QLoRALinear: QuantizedLinear, LoRAConvertToLinear {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: QuantizedLinear
    ) {

        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)

        // start frozen except for the lora keys
        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    /// - ``LoRALinear/from(linear:rank:)``
    public static func from(linear: QuantizedLinear, rank: Int = 8) -> Linear {
        var (outputDimensions, inputDimensions) = linear.shape
        inputDimensions = inputDimensions * 32 / linear.bits
        return QLoRALinear(inputDimensions, outputDimensions, rank: rank, linear: linear)
    }

    /// Convert back into a fused `QuantizedLinear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:layers:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:layers:deQuantize:)``
    public func toLinear(deQuantize: Bool = false) -> Linear {
        // convert back into full weights
        let weight = dequantized(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)

        let loraB = (scale * loraB.T).asType(.float16)
        let loraA = loraA.T.asType(.float16)

        // convert back into quantized
        return QuantizedLinear(
            weight: weight + matmul(loraB, loraA), bias: bias, groupSize: groupSize, bits: bits)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}
