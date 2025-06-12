//
//  DoRA+Layers.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXLinalg
import MLXNN
import MLXRandom

/// Performs the forward pass for a DoRA linear layer.
private func forward(
    x: MLXArray, y: MLXArray,
    weight: MLXArray, bias: MLXArray?,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let z = matmul(matmul(x, loraA), loraB)
    var out = y + (scale * z).asType(x.dtype)

    let adapted = weight + matmul(scale * loraB.T, loraA.T)
    let denom = norm(adapted, axis: 1)
    out *= (magnitude / denom).asType(x.dtype)

    return if let bias {
        out + bias
    } else {
        out
    }
}

/// Fuses the base weights with the DoRA parameters.
private func fuse(
    weight: MLXArray,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let loraA = loraA.T.asType(weight.dtype)
    let loraB = (scale * loraB.T).asType(weight.dtype)

    var adapted = weight + matmul(loraB, loraA)
    let denom = norm(adapted, axis: 1)
    adapted *= (magnitude / denom).reshaped([-1, 1])

    return adapted
}

/// Filters out DoRA-specific parameters from a list of module keys.
private func filterFreezeKeys(from module: Module, keys: [String]?) -> [String] {
    return
        (keys
        ?? module.filterMap(filter: type(of: module).filterLocalParameters)
        .flattened()
        .map { $0.0 })
        .filter { !["lora_a", "lora_b", "m"].contains($0) }
}

/// Implementation of DoRA `Linear` replacement layer.
///
/// This layer implements DoRA (Weight-Decomposed Low-Rank Adaptation) for `Linear` layers.
///
/// ``QDoRALinear`` is the equivalent class for `QuantizedLinear`.
public class DoRALinear: Linear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.weight, axis: 1)

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    public static func from(linear: Linear, rank: Int = 8, scale: Float = 20.0) -> LoRALayer {
        if let linear = linear as? QuantizedLinear {
            QDoRALinear(linear: linear, rank: rank, scale: scale)
        } else {
            DoRALinear(linear: linear, rank: rank, scale: scale)
        }
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    public func fused() -> Module {
        Linear(
            weight: fuse(
                weight: weight, loraA: loraA, loraB: loraB, scale: scale, magnitude: magnitude),
            bias: bias
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = matmul(x, weight.T)
        return forward(
            x: x, y: y,
            weight: weight, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude
        )
    }
}

/// Implementation of DoRA `QuantizedLinear` replacement layer.
///
/// See ``DoRALinear`` (equivalent class for `Linear` layers) for more information.
///
/// ### See Also
/// - ``DoRALinear``
public class QDoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required public init(linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0) {
        let (outputDimensions, inputDimensions) = linear.shape
        let loraScale = 1 / sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.dequantizedWeight, axis: 1)

        super.init(
            weight: linear.weight, bias: linear.bias,
            scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits
        )

        freeze()
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        let keys = filterFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    public func fused() -> Module {
        QuantizedLinear(
            weight: fuse(
                weight: dequantizedWeight, loraA: loraA, loraB: loraB, scale: scale,
                magnitude: magnitude),
            bias: bias, groupSize: groupSize, bits: bits
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = quantizedMatmul(
            x, weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
        return forward(
            x: x, y: y,
            weight: dequantizedWeight, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude
        )
    }
}
