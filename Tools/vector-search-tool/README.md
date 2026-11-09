# vector-search-tool

A command-line tool demonstrating [mlx-turbovec-swift](https://github.com/offlyn-ai/mlx-turbovec-swift) — TurboQuant vector quantization with optional MLX GPU acceleration.

Unlike `embedder-tool` (which embeds text with MLX Embedders and stores raw vectors as JSON), this tool shows **compressed vector search**: 768-dim embeddings shrink from 3,072 bytes to 384 bytes (4-bit) while maintaining >90% recall@10.

### Building

Build the `vector-search-tool` scheme in Xcode, or use Swift Package Manager:

```bash
swift build --target vector-search-tool
```

### Running: Command Line

Use the `mlx-run` helper after building in Xcode:

```bash
./mlx-run vector-search-tool --gpu --dim 768 --count 10000 --queries 100 --k 10
```

Pass `--debug` after `mlx-run` to run the Debug configuration.

Enable MLX GPU acceleration with `--gpu`. Without it, TurboVec falls back to Accelerate CPU paths (useful for CI and environments without Metal).

Write a structured JSON report for LLM analysis:

```bash
./mlx-run vector-search-tool --gpu --json /tmp/turbovec-report.json
```

### Running: Xcode

Configure scheme arguments (Product > Scheme > Edit Scheme > Run > Arguments):

```
--gpu --dim 768 --count 10000 --queries 50 --k 10
```

Then press <kbd>⌘</kbd>+<kbd>R</kbd> to run.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 768 | Vector dimension (multiple of 8) |
| `--count` | 10000 | Vectors to index |
| `--queries` | 100 | Search queries to run |
| `--k` | 10 | Top-k neighbors |
| `--bits` | 4 | Quantization bit width (2, 3, or 4) |
| `--gpu` | off | Enable MLX GPU acceleration |
| `--json` | — | Save JSON benchmark report to path |

### Expected Output

```
▸ MLX GPU acceleration enabled
▸ Generating 10000 random unit vectors (d=768)...
▸ Indexing with 4-bit TurboQuant...
  ✓ Indexed in 842.3ms
  ✓ Compression: 8.0× (30720000 → 3840000 bytes)
▸ Running 100 searches (k=10)...
  ✓ Mean latency: 0.412ms
  ✓ P99 latency:  0.891ms
  ✓ QPS:          2427
▸ Measuring recall vs brute-force baseline...
  ✓ Recall@1:  0.9100
  ✓ Recall@10: 0.9450
  ✓ Brute-force mean: 12.340ms
  ✓ Speedup: 29.9×

Summary: 8.0× compression, R@1=0.9100, 29.9× faster than brute-force
```

See also:

- [mlx-turbovec-swift](https://github.com/offlyn-ai/mlx-turbovec-swift) — the TurboQuant library
- [embedder-tool](../embedder-tool/README.md) — MLX Embedders for text embedding + JSON index
- [MLX troubleshooting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/troubleshooting)
