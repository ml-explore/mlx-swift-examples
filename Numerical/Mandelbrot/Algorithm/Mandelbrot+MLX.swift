// Copyright © 2026 Apple Inc.

import Foundation
import IOSurface
import MLX

/// Straightforward implementation of Mandelbrot using MLX.
///
/// Advantages:
///
/// - inner loop is a straightforward implementation of the math
/// - computes every point at once
/// - very little code overhead for c, z (vs plain swift)
///
/// Disadvantages:
///
/// - writes counts maxIterations times
/// - the compute graph is not necessarily fused
/// - no early exit -- every pixel runs through all the calculations
public func computeMandelbrotMLX(configuration: Configuration) -> MLXArray {
    let w = configuration.width
    let h = configuration.height
    let maxIterations = configuration.maxIterations
    let radius = configuration.escapeRadius

    let x = linspace(configuration.xMin, configuration.xMax, count: w)
    let y = linspace(configuration.yMin, configuration.yMax, count: h).reshaped(h, 1)

    let c = (x + y.asImaginary())
    var z = zeros(c.shape, dtype: .complex64)
    var counts = zeros(c.shape, dtype: .int16)

    for _ in 0 ..< maxIterations {
        z = z * z + c
        let mask = abs(z) .< radius
        counts = counts + mask
    }

    return counts
}

/// Straightforward implementation of Mandelbrot using MLX
/// with compiled graph.
///
/// Advantages:
///
/// - inner loop is a straightforward implementation of the math
/// - computes every point at once
/// - very little code overhead for c, z (vs plain swift)
/// - inner loop is compiled -- operations are fused
///
/// Disadvantages:
///
/// - writes counts maxIterations times
/// - the compile step complicates the loop
/// - no early exit -- every pixel runs through all the calculations
///
/// Is compilation worth it?  In this case probably yes -- the operations are all
/// elementwise and the intermediate arrays can be elided.  For a slight
/// loss in readability you might see 3-4x performance gain (on my laptop,
/// for this particular algorithm).  Hot inner loops with elementwise operations
/// are good candidates.
public func computeMandelbrotMLXCompiled(configuration: Configuration) -> MLXArray {
    let w = configuration.width
    let h = configuration.height
    let maxIterations = configuration.maxIterations
    let radius = configuration.escapeRadius

    let x = linspace(configuration.xMin, configuration.xMax, count: w)
    let y = linspace(configuration.yMin, configuration.yMax, count: h).reshaped(h, 1)

    let c = x + y.asImaginary()
    var z = zeros(c.shape, dtype: .complex64)
    var counts = zeros(c.shape, dtype: .int16)

    func step(z: MLXArray, c: MLXArray, counts: MLXArray) -> (MLXArray, MLXArray) {
        let z = z * z + c
        let mask = abs(z) .< radius
        let counts = counts + mask
        return (z, counts)
    }
    let compiledStep = compile(step)

    for _ in 0 ..< maxIterations {
        (z, counts) = compiledStep(z, c, counts)
    }

    return counts
}

private let mandelbrotKernel = MLXFast.metalKernel(
    name: "mandelbrot",
    inputNames: ["params"],
    outputNames: ["out"],
    source: """
        uint elem = thread_position_in_grid.x;
        int width = int(params[0]);
        int height = int(params[1]);
        int maxIterations = int(params[2]);
        float xMin = params[3];
        float yMin = params[4];
        float xStep = params[5];
        float yStep = params[6];
        float radiusSquared = params[7];

        if (elem >= uint(width * height)) return;

        int px = int(elem) % width;
        int py = int(elem) / width;

        float cReal = xMin + float(px) * xStep;
        float cImag = yMin + float(py) * yStep;

        float zReal = 0.0f;
        float zImag = 0.0f;
        int count = maxIterations;
        for (int i = 0; i < maxIterations; i++) {
            float zRealNew = zReal * zReal - zImag * zImag + cReal;
            zImag = 2.0f * zReal * zImag + cImag;
            zReal = zRealNew;
            if (zReal * zReal + zImag * zImag > radiusSquared) {
                count = i;
                break;
            }
        }
        out[elem] = short(count);
        """
)

/// Metal kernel implementation.
///
/// MLX also makes it (relatively) easy to define custom metal kernels.
/// The code isn't as easy to read as the plain MLX version, but for
/// this loop there are some real advantages.
///
/// Advantages:
///
/// - computes every point at once
/// - writes counts only once (likely the dominant win here)
/// - early exit per pixel, modulo SIMD/warp divergence (see below)
/// - it is fast!
///
/// Disadvantages:
///
/// - at least as much bookkeeping as the plain swift code
/// - writing in another language
///
/// Some algorithms will be much more efficient written in Metal.
/// In this case it can use a local variable for the count and avoid
/// reading and writing count for each iteration.  On my laptop
/// this runs roughly 10x faster than the compiled MLX code.
/// This can be worthwhile for some algorithms, but use
/// judiciously because it is more work to write and maintain.
///
/// A note on early exit: threads run in SIMD groups, so a pixel that escapes
/// quickly still waits for the slowest pixel in its group before the group
/// can retire.  When escape dominates a region you can observe roughly a 2x
/// speedup from the early exit, but the effect is harder to reason about
/// than the per-pixel early exit on the CPU.
public func computeMandelbrotMetal(configuration: Configuration) -> MLXArray {
    let w = configuration.width
    let h = configuration.height

    let params = MLXArray([
        Float(w),
        Float(h),
        Float(configuration.maxIterations),
        configuration.xMin,
        configuration.yMin,
        configuration.xStep,
        configuration.yStep,
        configuration.escapeRadiusSquared,
    ])

    let total = w * h
    let threadGroupSize = 256

    return mandelbrotKernel(
        [params],
        grid: (total, 1, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [[h, w]],
        outputDTypes: [.int16]
    )[0]
}

public func renderMandelbrotMLX(
    configuration: Configuration,
    compute: (Configuration) -> MLXArray = computeMandelbrotMetal
) -> IOSurface {
    let result = compute(configuration)
    let mlxLut = MLXArray(lut)
    let bgra = applyLUT(
        result, lut: mlxLut, max: Float(configuration.maxIterations), maxValue: 0xFF00_0000)
    return createIOSurface(bgra: bgra)
}
