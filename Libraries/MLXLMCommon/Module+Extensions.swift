// Copyright © 2024 Apple Inc.

import MLXNN

extension Module {

    /// Compute the number of parameters in a possibly quantized model
    public func numParameters() -> Int {
        return leafModules().flattenedValues().map {
            mod -> Int in
            if let qlin = mod as? QuantizedLinear {
                return qlin.scales.size * qlin.groupSize
            } else if let qemb = mod as? QuantizedEmbedding {
                return qemb.scales.size * qemb.groupSize
            } else {
                return mod.parameters().flattenedValues().reduce(
                    0,
                    {
                        $0 + $1.size
                    })
            }
        }.reduce(0, +)
    }
}
