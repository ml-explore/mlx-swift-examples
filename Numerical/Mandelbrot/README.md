# Mandelbrot

Renders the **Mandelbrot set** by iterating `z ← z² + c` for every pixel
in parallel and coloring by escape time. Three implementations are
provided so you can compare:

| Implementation | Where | Notes |
|---|---|---|
| Plain MLX | [`computeMandelbrotMLX`](Algorithm/Mandelbrot+MLX.swift) | Straightforward — uses `complex64` and `linspace` to build `c`, then loops the recurrence over the whole grid. |
| Compiled MLX | [`computeMandelbrotMLXCompiled`](Algorithm/Mandelbrot+MLX.swift) | Same math, wrapped in `compile(...)`. Operations fuse; ~3–4× faster than plain MLX on the inner loop. |
| Metal kernel | [`computeMandelbrotMetal`](Algorithm/Mandelbrot+MLX.swift) | Custom `MLXFast.metalKernel`. Counts live in a local variable (no per-iteration writes) and pixels can early-exit. ~10× faster than the compiled MLX version. |
| Reference CPU | [`Mandelbrot+CPU.swift`](Algorithm/Mandelbrot+CPU.swift) | Plain Swift, for comparison and correctness. |

Each algorithm returns an `MLXArray` of escape counts; the shared
[`renderMandelbrotMLX`](Algorithm/Mandelbrot+MLX.swift) applies a color LUT
and produces an `IOSurface` for display.

## Algorithm

[`Algorithm/`](Algorithm)

- [`Mandelbrot+MLX.swift`](Algorithm/Mandelbrot+MLX.swift) — the three MLX
  implementations described above. The doc comments on each function lay
  out the trade-offs.
- [`Configuration.swift`](Algorithm/Configuration.swift) — image size,
  iteration cap, viewport (`centerX`/`centerY`/`zoom`), color LUT, and a
  `lerp(other:steps:step:)` helper used for animated zooms.

## UI

[`ContentView.swift`](ContentView.swift) drives
[`Renderer.swift`](Renderer.swift) and lets you pick which implementation
to run. A frame-time readout shows the wall-clock difference between the
three.
