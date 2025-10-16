# embedder-tool

See additional documentation for supporting libraries:

- [Embedders](../../Libraries/Embedders/README.md)

### Building

Build the `embedder-tool` scheme in Xcode.

### Running: Xcode

Configure the scheme arguments (Product > Scheme > Edit Scheme > Run > Arguments). For example, to build an index of Markdown files in `Libraries`:

```
index \
    --output /tmp/embedder-index.json \
    --directory Libraries \
    --extensions md \
    --recursive
```

Then press <kbd>⌘</kbd>+<kbd>R</kbd> to run. The first launch may prompt for access to the Documents directory so the Hugging Face `HubApi` can download model assets.

### Running: Command Line

Use the `mlx-run` helper to locate the built binary:

```
./mlx-run embedder-tool index --output /tmp/embedder-index.json --directory Libraries --extensions md --recursive
```

By default this runs the Release build. Pass `--debug` after `mlx-run` to execute the Debug configuration.

### Commands

`embedder-tool` defaults to the `nomic-ai/nomic-embed-text-v1.5` configuration but any registered model from `embedder-tool list` (or a local directory) can be selected with `--model`. Download locations can be overridden with `--download`. 

Pooling defaults to mean strategy with normalization enabled, which can be customized with `--strategy`, `--normalize`, and `--layer-norm`. Passing `--no-normalize` keeps pooled vectors at their raw magnitudes across indexing, search, and REPL flows while replacing any non-finite components with zero.

#### Common options

- **Model selection**: `--model` accepts a registered configuration name or a path to a local directory. Use `--download` to point the Hugging Face cache at a custom location.
- **Pooling**: Choose a pooling strategy with `--strategy` (`mean`, `cls`, `first`, `last`, `max`, or `none`). `--no-normalize` skips L2 normalization and `--layer-norm` applies layer normalization before pooling.
- **Corpus**: Commands that load documents accept `--directory`, `--extensions` (defaults to `txt md`), `--recursive`, and `--limit` to cap how many files are embedded.

#### index

Creates an embedding index for a corpus and writes it as prettified JSON.

```
./mlx-run embedder-tool index \
    --output /tmp/embedder-index.json \
    --directory Data/corpus \
    --extensions md txt \
    --recursive \
    --batch-size 8
```

Each document in the target directory (filtered by `--extensions`, optionally `--recursive`, and `--limit`) is embedded in batches controlled by `--batch-size`. The default is 8 to keep memory usage modest; raise it (for example, `--batch-size 32`) when your hardware has room and you want higher throughput.

Vectors are sanitized before writing so NaN/Inf values become zero. When invoked with `--no-normalize`, the stored vectors keep their original magnitudes.

#### search

Embeds a query and reports cosine similarity scores against an existing index. When the index was built with `--no-normalize`, use the same flag so query vectors stay unnormalized and dot products reflect magnitude as well as direction. Query vectors are sanitized before scoring to avoid NaN/Inf propagation.

```
./mlx-run embedder-tool search \
    --index /tmp/embedder-index.json \
    --query "swift embeddings" \
    --top 5
```

Results whose vector dimensions mismatch the query are skipped with warnings, and any pooling fallbacks are reported.

#### repl

Builds an in-memory embedding index for a directory and launches a simple REPL for quick experiments. Press return on an empty line to exit the loop.

```
./mlx-run embedder-tool repl \
    --directory Data/corpus \
    --extensions md txt \
    --recursive \
    --top 5 \
    --show-timing
```

Use `/help` inside the REPL to discover commands like `/stats` and `/quit`.

`--batch-size` (default 8) controls how many documents are embedded per pass, `--top` limits the number of matches shown, and `--show-timing` prints millisecond timings. As with `index` and `search`, pass `--no-normalize` to work with raw magnitudes.

#### list

Lists the registered embedder configurations. Add `--include-directories` to show locally registered directories.

```
./mlx-run embedder-tool list --include-directories
```

#### demo

Runs a sample workflow that indexes documentation from the repository and queries it.

```
./mlx-run embedder-tool demo --keep-index
```

Provide one or more queries as positional arguments to override the defaults:

```
./mlx-run embedder-tool demo "How to train a model?"
```

When `--keep-index` is omitted, the temporary index is deleted after the demo finishes.

The demo command reuses the `index` and `search` flows under the hood, limiting the corpus to a handful of Markdown files and running either the default sample queries or the queries you supply.
