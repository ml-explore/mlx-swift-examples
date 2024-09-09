// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/sampler.py

/// Interpolate the function defined by `(0 ..< y.count) y)` at positions `xNew`.
func interpolate(y: MLXArray, xNew: MLXArray) -> MLXArray {
    let xLow = xNew.asType(.int32)
    let xHigh = minimum(xLow + 1, y.count - 1)

    let yLow = y[xLow]
    let yHigh = y[xHigh]
    let deltaX = xNew - xLow
    let yNew = yLow * (1 - deltaX) + deltaX * yHigh

    return yNew
}

/// A simple Euler integrator that can be used to sample from our diffusion models.
///
/// The method ``step()`` performs one Euler step from `x_t` to `x_t_prev`.
class SimpleEulerSampler {

    let sigmas: MLXArray

    public init(configuration: DiffusionConfiguration) {
        let betas: MLXArray

        // compute the noise schedule
        switch configuration.betaSchedule {
        case .linear:
            betas = MLXArray.linspace(
                configuration.betaStart, configuration.betaEnd, count: configuration.trainSteps)
        case .scaledLinear:
            betas = MLXArray.linspace(
                sqrt(configuration.betaStart), sqrt(configuration.betaEnd),
                count: configuration.trainSteps
            ).square()
        }

        let alphas = 1 - betas
        let alphasCumprod = cumprod(alphas)

        self.sigmas = concatenated([
            MLXArray.zeros([1]), ((1 - alphasCumprod) / alphasCumprod).sqrt(),
        ])
    }

    public var maxTime: Int {
        sigmas.count - 1
    }

    public func samplePrior(shape: [Int], dType: DType = .float32, key: MLXArray? = nil) -> MLXArray
    {
        let noise = MLXRandom.normal(shape, key: key)
        return (noise * sigmas[-1] * (sigmas[-1].square() + 1).rsqrt()).asType(dType)
    }

    public func addNoise(x: MLXArray, t: MLXArray, key: MLXArray? = nil) -> MLXArray {
        let noise = MLXRandom.normal(x.shape, key: key)
        let s = sigmas(t)
        return (x + noise * s) * (s.square() + 1).rsqrt()
    }

    public func sigmas(_ t: MLXArray) -> MLXArray {
        interpolate(y: sigmas, xNew: t)
    }

    public func timeSteps(steps: Int, start: Int? = nil, dType: DType = .float32) -> [(
        MLXArray, MLXArray
    )] {
        let start = start ?? (sigmas.count - 1)
        precondition(0 < start)
        precondition(start <= sigmas.count - 1)
        let steps = MLX.linspace(start, 0, count: steps + 1).asType(dType)

        return Array(zip(steps, steps[1...]))
    }

    open func step(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray) -> MLXArray {
        let dtype = epsPred.dtype
        let sigma = sigmas(t).asType(dtype)
        let sigmaPrev = sigmas(tPrev).asType(dtype)

        let dt = sigmaPrev - sigma
        var xtPrev = (sigma.square() + 1).sqrt() * xt + epsPred * dt
        xtPrev = xtPrev * (sigmaPrev.square() + 1).rsqrt()

        return xtPrev
    }
}

class SimpleEulerAncestralSampler: SimpleEulerSampler {

    open override func step(epsPred: MLXArray, xt: MLXArray, t: MLXArray, tPrev: MLXArray)
        -> MLXArray
    {
        let dtype = epsPred.dtype
        let sigma = sigmas(t).asType(dtype)
        let sigmaPrev = sigmas(tPrev).asType(dtype)

        let sigma2 = sigma.square()
        let sigmaPrev2 = sigmaPrev.square()
        let sigmaUp = (sigmaPrev2 * (sigma2 - sigmaPrev2) / sigma2).sqrt()
        let sigmaDown = (sigmaPrev2 - sigmaUp ** 2).sqrt()

        let dt = sigmaDown - sigma
        var xtPrev = (sigma2 + 1).sqrt() * xt + epsPred * dt
        let noise = MLXRandom.normal(xtPrev.shape).asType(xtPrev.dtype)
        xtPrev = xtPrev + noise * sigmaUp
        xtPrev = xtPrev * (sigmaPrev2 + 1).rsqrt()

        return xtPrev
    }
}
