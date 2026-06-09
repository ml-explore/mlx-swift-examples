// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Quadratic model `θ₀ + θ₁·x + θ₂·x²` — the function we are fitting.
func model(_ θ: MLXArray, _ x: MLXArray) -> MLXArray {
    θ[0] + θ[1] * x + θ[2] * x * x
}

/// The target function we are trying to recover. The model is intentionally
/// under-parameterized (quadratic vs. cubic) so the fit is imperfect.
func target(_ x: MLXArray) -> MLXArray {
    3 + x * 0.5 + 3 * x * x - x * x * x
}

/// Fits ``model`` to noisy samples of ``target`` via gradient descent.
struct Gradient {

    let numParams = 3
    let totalSteps = 50
    let learningRate: Float = 0.005

    let x: MLXArray
    let y: MLXArray
    var θ: MLXArray

    private let gradLoss: (MLXArray) -> MLXArray

    init() {
        let x = MLX.linspace(Float(-2.0), Float(2.0), count: 40)

        // target function + noise
        let y = target(x) + MLXRandom.uniform(Float(-1) ..< Float(1), x.shape)

        self.x = x
        self.y = y
        self.θ = zeros([numParams])

        func loss(_ θ: MLXArray) -> MLXArray {
            mean((model(θ, x) - y) ** 2)
        }
        self.gradLoss = grad(loss)
    }

    mutating func step() {
        let g = gradLoss(θ)  // ∇L(θ)
        θ = θ - learningRate * g  // parameter update
        eval(θ)
    }
}
