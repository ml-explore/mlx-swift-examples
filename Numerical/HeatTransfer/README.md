# Jacobi

Simulates **heat diffusion** in a 2D room with walls and fixed heat
sources. Each step replaces every cell with the average of its four
neighbors (the discrete Laplacian) — heat sources stay pinned, walls are
held fixed, and the rest of the grid relaxes toward equilibrium.

Three implementations let you compare how the same stencil computation
can be expressed in MLX:

| Implementation | Where | Notes |
|---|---|---|
| `conv2d` | [`computeJacobiConv2d`](Algorithm/Jacobi+MLX.swift) | A 3×3 kernel with weights on the four neighbors. Reads more points than strictly needed but the convolution path is highly optimized. |
| `roll` (compiled) | [`computeJacobiStencil`](Algorithm/Jacobi+MLX.swift) | Builds the four-point stencil directly by shifting the grid with `roll(...)`. Wrapped in `compile(...)` so the shifts and add fuse — ~2–3× faster than `conv2d` on a laptop. |
| Successive over-relaxation | [`computeSOR`](Algorithm/Jacobi+MLX.swift) | Different *algorithm* — a red/black SOR sweep with optimal ω. Per-step cost is ~2× the Jacobi step, but it converges in O(n) sweeps instead of O(n²). |

The simulation runs many iterations per displayed frame (200 by default
for the Jacobi variants; SOR can be run either at 1/frame to *see* it
converge or 200/frame for throughput).

## Algorithm

[`Algorithm/`](Algorithm)

- [`Jacobi+MLX.swift`](Algorithm/Jacobi+MLX.swift) — the three
  implementations. Doc comments on each function describe the trade-offs
  and the `checkerboard(...)` helper used by SOR.
- [`Configuration.swift`](Algorithm/Configuration.swift) — room layout
  (walls, heat sources, masks) and the `asRoom()` builder that turns a
  configuration into the `MLXArray`s the algorithms consume.

## UI

[`ContentView.swift`](ContentView.swift) drives
[`Renderer.swift`](Renderer.swift). The renderer lets you pick a
variant (`Conv2D`, `Roll`, `SOR`, `SOR Full Speed`), randomize the room,
and reports per-iteration timing. Note: the timing window calls
`eval(room.temperature)` to force GPU sync — without it, MLX's lazy
evaluation would defer the work past the timer.
