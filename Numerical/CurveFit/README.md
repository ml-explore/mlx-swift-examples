# Gradient

A live visualization of **gradient descent** fitting a quadratic model
`θ₀ + θ₁·x + θ₂·x²` to noisy samples drawn from a cubic target. Each step
the parameters `θ` move down the gradient of mean-squared-error loss; the
fitted curve is replotted alongside the data so you can watch convergence.

The app is built around `MLX.grad`, which produces the gradient function for
`loss(θ)` automatically — no manual derivatives.

## Algorithm

[`Algorithm/Gradient.swift`](Algorithm/Gradient.swift)

- `model(_:_:)` — the quadratic being fit.
- `target(_:)` — the cubic ground truth (intentionally one degree higher than
  the model, so the fit is imperfect).
- `Gradient.init()` — samples the target with uniform noise and builds
  `gradLoss = grad(loss)`.
- `Gradient.step()` — one update: `θ ← θ − η · ∇L(θ)`, then `eval(θ)`.

## UI

[`ContentView.swift`](ContentView.swift) — Swift Charts plots the actual
samples in blue and the model's prediction in orange. **Start** runs
`totalSteps` updates with a fixed delay between frames; **Reset** redraws
fresh noisy samples and zeros `θ`.
