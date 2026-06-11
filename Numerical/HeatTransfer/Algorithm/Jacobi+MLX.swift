// Copyright © 2026 Apple Inc.

import Foundation
import IOSurface
import MLX

/// Straightforward implementation of Jacobi iterations using
/// `conv2d` -- the weights implement the stencil.
///
/// Advantages:
///
/// - inner loop is a straightforward implementation of the math
/// - computes every point at once
/// - convolution is likely highly optimized
///
/// Disadvantages:
///
/// - convolution does read 9 points when it only needs 4
///
/// Note: the computation is very fast -- much faster than the 60 Hz rate we are targeting
/// for the display.  We use the `count` to run multiple iterations per displayed frame.
/// This speeds up the display (more interesting for humans) and amortizes the overhead
/// a bit.
public func computeJacobiConv2d(state: inout Room, count: Int) {
    // [1, H, W, 1] -- match what conv2d needs
    var temperature = state.temperature[.newAxis, .ellipsis, .newAxis]
    let staticMask = state.staticMask[.newAxis, .ellipsis, .newAxis]
    let heatSources = state.heatSources[.newAxis, .ellipsis, .newAxis]

    let kernel = MLXArray(converting: [
        0, 0.25, 0,
        0.25, 0, 0.25,
        0, 0.25, 0,
    ]
    ).reshaped(1, 3, 3, 1)

    for _ in 0 ..< count {
        let next = conv2d(temperature, kernel, padding: 1)
        temperature = which(staticMask, heatSources, next)
    }

    state.temperature = temperature.squeezed()
}

/// A more direct implementation of the 4 point stencil.
///
/// If you were writing this for the CPU it would look something like this (inner loop, per point):
///
/// ```swift
/// next[x, y] = 0.25 * (current[x - 1, y] + current[x + 1, y] + current[x, y - 1] + current[x, y + 1])
/// ```
///
/// In array processing code you would do the same thing by _moving_ the array
/// around and then doing element-wise arithmetic.  This is not (typically) physically
/// shifting the array around, but manipulating a view on top of it.
///
/// We can shift the array around using:
///
/// - `roll()` -- rotate the array on one axis.  this works because there are walls on the border
/// - slicing -- e.g. indexing operations like `temperature[1..., 0...]` (shift up).  This
/// requires dealing with losing an edge value (the shape is smaller by 2 in each direction)
/// - `padded()` + slicing -- same as slicing, but add padding to fix the edge issue
///
/// This example shows roll.
///
/// Advantages:
///
/// - inner loop is recognizable compared to the math
/// - computes every point at once
/// - does less work than convolution
/// - amenable to `compile()` to fuse the operations
///
/// Disadvantages:
///
/// - the code exposes more of the details -- it is slightly harder to read than convolution
/// - the edge conditions are a real concern and the simplicity of this problem largely avoids them
///
/// On my laptop I find that this is 2 to 3 times faster than conv2d when compiled.  With roll
/// you have to reason about the edges explicitly. Convolution handles them via padding.
///
/// Note: the computation is very fast -- much faster than the 60 Hz rate we are targeting
/// for the display.  We use the `count` to run multiple iterations per displayed frame.
/// This speeds up the display (more interesting for humans) and amortizes the overhead
/// a bit.
public func computeJacobiStencil(state: inout Room, count: Int) {

    @Sendable
    func step(_ temperature: MLXArray, _ staticMask: MLXArray, _ heatSources: MLXArray) -> MLXArray
    {
        let next =
            0.25
            * (roll(temperature, shift: -1, axis: 0)
                + roll(temperature, shift: 1, axis: 0)
                + roll(temperature, shift: -1, axis: 1)
                + roll(temperature, shift: 1, axis: 1))
        return which(staticMask, heatSources, next)
    }
    let compiledStep = compile(step)

    for _ in 0 ..< count {
        state.temperature = compiledStep(
            state.temperature, state.staticMask, state.heatSources)
    }
}

func checkerboard(rows: Int, cols: Int, phase: Int) -> MLXArray {
    let rows = arange(0, rows)[.ellipsis, .newAxis]
    let cols = arange(0, cols)[.newAxis, .ellipsis]

    return (((rows + cols) % 2) .== phase)[.newAxis, .ellipsis, .newAxis]
}

/// A more efficient algorithm for computing heat transfer.
///
/// Jacobi iterations takes O(n^2) (where n is the length of
/// the edge of the grid) iterations to converge.
/// Successive over-relaxation (SOR)
/// converges in O(n) when an optimal ω is used.  This is
/// like switching bubble sort for quicksort.
///
/// This uses an ω to overdrive the prediction. Red/black lets each color's
/// update read the most recent values from its opposite-color neighbors,
/// giving Gauss-Seidel-style propagation without serialising the grid.
///
/// Advantages:
///
/// - inner loop is a straightforward implementation of the math
/// - computes every point at once
/// - big win: this converges faster
///
/// Disadvantages:
///
/// - the math is more complex
/// - this is twice as expensive as conv2d (it uses 2 of them) but fewer calls are needed
///
/// When comparing the amortized version (200 iterations, see "SOR Full Speed"), this is
/// roughly twice as slow as conv2d.  SOR (1 iter/frame) runs ~10x slower per
/// iteration than SOR Full Speed (200 iter/frame) — dispatch and sync overhead
/// dominate at small batch sizes.
///
/// Note: the computation is very fast -- much faster than the 60 Hz rate we are targeting
/// for the display.  We use the `count` to run multiple iterations per displayed frame.
/// This speeds up the display (more interesting for humans) and amortizes the overhead
/// a bit.
public func computeSOR(state: inout Room, count: Int) {
    var temperature = state.temperature[.newAxis, .ellipsis, .newAxis]
    let heatMask = state.staticMask[.newAxis, .ellipsis, .newAxis]
    let heatSources = state.heatSources[.newAxis, .ellipsis, .newAxis]

    let M = state.temperature.dim(0)
    let N = state.temperature.dim(1)

    let kernel = MLXArray(converting: [
        0, 0.25, 0,
        0.25, 0, 0.25,
        0, 0.25, 0,
    ]
    ).reshaped(1, 3, 3, 1)

    let ω: Float = 2.0 / (1.0 + sin(Float.pi / Float(max(M, N))))

    let redMask = checkerboard(rows: M, cols: N, phase: 0)
    let blackMask = checkerboard(rows: M, cols: N, phase: 1)

    for _ in 0 ..< count {
        // Update red cells using black neighbors
        let sorRed = ω * conv2d(temperature, kernel, padding: 1) + (1 - ω) * temperature
        temperature = which(redMask, sorRed, temperature)
        temperature = which(heatMask, heatSources, temperature)

        // Update black cells using (now-updated) red neighbors
        let sorBlack = ω * conv2d(temperature, kernel, padding: 1) + (1 - ω) * temperature
        temperature = which(blackMask, sorBlack, temperature)
        temperature = which(heatMask, heatSources, temperature)
    }

    state.temperature = temperature.squeezed()
}
