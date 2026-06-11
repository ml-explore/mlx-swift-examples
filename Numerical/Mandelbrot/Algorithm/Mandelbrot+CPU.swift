// Copyright © 2026 Apple Inc.

import ComplexModule
import Foundation
import IOSurface

/// Compute Mandelbrot set using standard Swift Float
public func computeMandelbrotCPUFloat(configuration: Configuration) -> Array2D<Int> {
    let w = configuration.width
    let h = configuration.height
    let maxIterations = configuration.maxIterations

    let xMin = configuration.xMin
    let yMin = configuration.yMin
    let xStep = configuration.xStep
    let yStep = configuration.yStep
    let radiusSquared = configuration.escapeRadiusSquared

    var counts = Array2D<Int>(width: w, height: h)

    for y in 0 ..< h {
        for x in 0 ..< w {
            let cReal = xMin + Float(x) * xStep
            let cImag = yMin + Float(y) * yStep
            var zReal: Float = 0
            var zImag: Float = 0
            var limit = maxIterations
            for i in 0 ..< maxIterations {
                let zRealNew = zReal * zReal - zImag * zImag + cReal
                zImag = 2 * zReal * zImag + cImag
                zReal = zRealNew
                let magnitudeSquared = zReal * zReal + zImag * zImag
                if magnitudeSquared > radiusSquared {
                    limit = i
                    break
                }
            }
            counts[x, y] = limit
        }
    }

    return counts
}

/// Compute Mandelbrot set using swift-numerics Complex.
///
/// The inside of the inner loop is a straightforward implementation of
/// the Mandelbrot algorithm -- the Complex type abstracts the math
/// we are trying to express.
///
/// Advantages:
///
/// - it stops early -- for points that escape quickly the loop can exit early
/// - it only writes counts once
///
/// Disadvantages:
///
/// - ceremony building c, z
/// - computes one point at a time
///
/// The two advantages are major wins for the CPU -- you can see this effect on areas
/// where the escaping points dominate.  The render time might improve by 10x or
/// more, depending which part of the fractal is rendered.
public func computeMandelbrotCPUComplex(configuration: Configuration) -> Array2D<Int> {
    let w = configuration.width
    let h = configuration.height
    let maxIterations = configuration.maxIterations

    let xMin = configuration.xMin
    let yMin = configuration.yMin
    let xStep = configuration.xStep
    let yStep = configuration.yStep
    let radiusSquared = configuration.escapeRadiusSquared

    var counts = Array2D<Int>(width: w, height: h)

    for y in 0 ..< h {
        for x in 0 ..< w {
            let c = Complex(xMin + Float(x) * xStep, yMin + Float(y) * yStep)
            var z = Complex<Float>.zero
            var limit = maxIterations
            for i in 0 ..< maxIterations {
                z = z * z + c
                if z.lengthSquared > radiusSquared {
                    limit = i
                    break
                }
            }
            counts[x, y] = limit
        }
    }

    return counts
}

public func renderMandelbrotCPU(configuration: Configuration) -> IOSurface {
    let result = computeMandelbrotCPUComplex(configuration: configuration)
    let bgra = applyLUT(
        result, lut: lut, max: Float(configuration.maxIterations), maxValue: 0xFF00_0000)
    return createIOSurface(bgra: bgra)
}
