# MuseGlimmerDemo

Runs [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
entirely on-device: drop in an image, ask about it, watch tokens stream. After the
initial model download there are no network calls.

Build and run the `MuseGlimmerDemo` scheme. **Use the Release configuration** — a
Debug build of a 30B forward pass is several times slower (first token takes ~21 s
in Debug versus ~10 s in Release on the same image).

## Requirements

`mlx-community/Muse-Glimmer-30B-4bit` is a 19.4 GB download and sits around
**19 GB resident** — roughly 3.7 GB of that is the vision tower, which the
checkpoint leaves unquantized in bf16. A 64 GB machine is comfortable; 32 GB is
not. Measured on an M4 Max: ~26 tok/s decode, 18.97 GB peak.

The app raises the buffer cache to 2 GB (`Memory.cacheLimit`) in
`MuseGlimmerService.configureMemory()`. The 20 MB the smaller examples use
thrashes badly at this size, because a single image encode churns hundreds of MB
of activations.

## Why there is a wait before the first token

Almost all of the perceived latency is prefill, driven by how many tokens the
image expands into — not by decode, which is a steady ~26 tok/s. The app reports
the phase, a prefill progress bar, the prompt composition and time-to-first-token
so the wait reads as work rather than a hang.

Measured on an M4 Max (Release):

| input | prompt tokens | ViT patches | time to first token |
|---|---|---|---|
| 448×448 image | 318 | 1,024 | 1.8 s |
| 768×1024 image | 1,034 | 3,888 | 5.8–6.6 s |
| 2400×2400 image | 4,158 | 16,384 | 29–40 s |
| text only | 4,205 | — | 19–20 s |

Text prefill runs at ~200 tok/s and is the dominant term: an image doesn't just
cost ViT time, it injects up to 4096 tokens into the prompt and every one goes
through the 52-layer text stack. Vision encode adds ~1.5 s at 3,888 patches but
10–20 s at 16,384, superlinear because the ViT's 13 full-attention layers are
O(L²) in the *unmerged* patch count.

The **Image budget** slider is therefore the strongest latency control. It
defaults to 1024 merged tokens rather than the checkpoint's 4096, which on the
2400×2400 sample takes first token from 26.6 s to 7.2 s. It routes through
`UserInput.Processing.maxPixels`, where one merged token covers 28×28 pixels.

## Notes

- **Reasoning is not observable.** The model reasons under
  `to=self<|message|>…<|eom|>` before answering under `to=user<|message|>…`. The
  framework strips those control tokens and withholds reasoning from the public
  `Generation` stream, and `TokenStreamDecoder` is `package`-level, so an app
  cannot consume it. The effect is a second silent window after prefill — on the
  sample image prefill finishes at 5.8 s but the first visible token lands at
  10.5 s. The status strip names that window rather than leaving the pane blank.
- **Greedy decoding** (`temperature: 0`) so output can be compared against the
  Python reference implementation.
- Single image per turn. Video is rejected explicitly; tool calling and
  multi-image prompts are not wired up in this app.
- An image path passed as a launch argument preloads the well, which makes the
  drop-to-describe flow scriptable. Under the app sandbox only user-selected
  files are readable, so use drag-and-drop or the file picker normally.
