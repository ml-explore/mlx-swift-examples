// Copyright © 2026 Apple Inc.

import CoreVideo
import Foundation
import IOSurface
import MLX

public func applyLUT(_ input: MLXArray, lut: MLXArray, max: Float, maxValue: UInt32) -> MLXArray {
    precondition(lut.ndim == 1)
    let lutCount = lut.dim(0)

    // LUT is BGRA and we want to interpolate per channel
    let lut = lut.view(dtype: .uint8)
        .reshaped([lutCount, 4])
        .asType(.float32)

    // interpolate 0 ... max -> lut indexes
    let scale = (Float(lutCount) - 1) / max
    let index = input.asType(.float32) * scale

    let lutIndexLow = floor(index).asType(.int16)
    let lutIndexHigh = minimum(lutIndexLow + 1, lutCount - 1)

    // compute the fraction between the lut values.
    // add .newAxis so that it will broadcast for the channels
    let fraction = (index - lutIndexLow)[.ellipsis, .newAxis]

    let colorLow = lut.take(lutIndexLow, axis: 0)
    let colorHigh = lut.take(lutIndexHigh, axis: 0)

    // the produces [H, W, 4]
    let maxValue = MLXArray([maxValue]).view(dtype: .uint8)
    let result = which(
        input[.ellipsis, .newAxis] .>= max, maxValue, colorLow + fraction * (colorHigh - colorLow))
    return round(result).asType(.uint8)
}

public func createIOSurface(bgra: MLXArray) -> IOSurface {
    precondition(bgra.ndim == 3)
    precondition(bgra.dim(2) == 4)
    precondition(bgra.dtype == .uint8)

    // Zero-copy access to MLX backing data
    let arrayData = bgra.asData(access: .noCopyIfContiguous)

    let w = bgra.dim(1)
    let h = bgra.dim(0)

    // Create IOSurface and memcpy
    let bytesPerRow = w * 4
    let surface = IOSurface(properties: [
        .width: w,
        .height: h,
        .bytesPerElement: 4,
        .bytesPerRow: bytesPerRow,
        .pixelFormat: kCVPixelFormatType_32BGRA,
    ])!

    surface.lock(options: [], seed: nil as UnsafeMutablePointer<UInt32>?)
    _ = arrayData.data.withUnsafeBytes { src in
        memcpy(surface.baseAddress, src.baseAddress!, h * bytesPerRow)
    }
    surface.unlock(options: [], seed: nil as UnsafeMutablePointer<UInt32>?)

    return surface
}
