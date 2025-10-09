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
    --recursive \
    --normalize
```

Then press <kbd>⌘</kbd>+<kbd>R</kbd> to run. The first launch may prompt for access to the Documents directory so the Hugging Face `HubApi` can download model assets.

### Running: Command Line

Use the `mlx-run` helper to locate the built binary:

```
./mlx-run embedder-tool index --output /tmp/embedder-index.json --directory Libraries --extensions md --recursive --normalize
```

By default this runs the Release build. Pass `--debug` after `mlx-run` to execute the Debug configuration.

### Commands

`embedder-tool` defaults to the `nomic-ai/nomic-embed-text-v1.5` configuration but any registered model from `embedder-tool list` (or a local directory) can be selected with `--model`. Download locations can be overridden with `--download`. Pooling behavior is configured with `--strategy`, `--normalize`, and `--layer-norm`.

#### index

Creates an embedding index for a corpus and writes it as prettified JSON.

```
./mlx-run embedder-tool index \
    --output /tmp/embedder-index.json \
    --directory Data/corpus \
    --extensions md txt \
    --recursive \
    --batch-size 32 \
    --normalize
```

Each document in the target directory (filtered by `--extensions`, optionally `--recursive`, and `--limit`) is embedded in batches controlled by `--batch-size`.

#### search

Embeds a query and reports cosine similarity scores against an existing index.

```
./mlx-run embedder-tool search \
    --index /tmp/embedder-index.json \
    --query "swift embeddings" \
    --top 5 \
    --normalize
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
    --normalize \
    --show-timing
```

Use `/help` inside the REPL to discover commands like `/stats` and `/quit`.

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
