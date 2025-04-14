# llm-tool

See various READMEs:

- [MLXLMCommon](../../Libraries/MLXLMCommon/README.md) -- common LM code
- [MLXLLM](../../Libraries/MLXLLM/README.md) -- large language models
- [MLXVLM](../../Libraries/MLXVLM/README.md) -- vision language models

### Building

Build the `llm-tool` scheme in Xcode.

### Running: Xcode

To run this in Xcode simply press cmd-opt-r to set the scheme arguments.  For example:

```
--model mlx-community/Mistral-7B-Instruct-v0.3-4bit
--prompt "swift programming language"
--max-tokens 50
```

Then cmd-r to run.

> Note: you may be prompted for access to your Documents directory -- this is where
the Hugging Face HubApi stores the downloaded files.

The model should be a path in the Hugging Face repository, e.g.:

- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/phi-2-hf-4bit-mlx`

See [LLM](../../Libraries/MLXLLM/README.md) for more info.

### Running: Command Line

Use the `mlx-run` script to run the command line tools:

```
./mlx-run llm-tool --prompt "swift programming language"
```

Note: `mlx-run` is a shell script that uses `xcode` command line tools to
locate the built binaries.  It is equivalent to running from Xcode itself.

By default this will find and run the tools built in _Release_ configuration.  Specify `--debug`
to find and run the tool built in _Debug_ configuration.

See also:

- [MLX troubleshooting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/troubleshooting)

### Troubleshooting

If the program crashes with a very deep stack trace you may need to build
in Release configuration.  This seems to depend on the size of the model.

There are a couple options:

- build Release
- force the model evaluation to run on the main thread, e.g. using @MainActor
- build `Cmlx` with optimizations by modifying `mlx/Package.swift` and adding `.unsafeFlags(["-O"]),` around line 87

Building in Release / optimizations will remove a lot of tail calls in the C++ 
layer.  These lead to the stack overflows.

See discussion here: https://github.com/ml-explore/mlx-swift-examples/issues/3

## LoRA

`llm-tool` provides an example LoRA driver based on:

- https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md

This is an example of using MLX to fine-tune an LLM with low rank adaptation
(LoRA) for a target task.[^lora] The example also supports quantized LoRA
(QLoRA).[^qlora] The example works with Llama and Mistral style models
available on Hugging Face.

In this example we'll use the WikiSQL[^wikisql] dataset to train the LLM to
generate SQL queries from natural language. However, the example is intended to
be general should you wish to use a custom dataset.

> Note: Some of the prompts have newlines in them which is difficult to achieve via running in Xcode.

Running `llm-tool lora` will produce help:

```
SUBCOMMANDS:
  train                   LoRA training
  fuse                    Fuse lora adapter weights back in to original model
  test                    LoRA testing
  eval                    LoRA evaluation
```

### Training

The first step will be training the LoRA adapter.  Example training data
is available in $SRCROOT/Data/lora.  You can use your
own data in either `jsonl` or `txt` format with one entry per line.

We need to specify a number of parameters:

- `--model` -- which model to use.  This can be quantized [^qlora] or not [^lora]
- `--data` -- directory with the test, train and valid files.  These can be either `jsonl` or `txt` files
- `--adapter` -- path to a safetensors file to write the fine tuned parameters into

Additionally the performance of the fine tuning can be controlled with:

- `--batch-size` -- size of the minibatches to run in the training loop, e.g. how many prompts to process per iteration
- `--lora-layers` -- the number of layers in the Attention section of the model to adapt and train
- `--iterations` -- the number of iterations to train for

If desired, the amount of memory used can be adjusted with:

- `--cache-size` -- the number shown below limits the cache size to 1024M 

Here is an example run using adapters on the last 4 layers of the model:

```
./mlx-run llm-tool lora train \
    --model mlx-community/Mistral-7B-v0.1-hf-4bit-mlx \
    --data Data/lora \
    --adapter /tmp/lora-layers-4.safetensors \
    --batch-size 1 --lora-layers 4 \
    --cache-size 1024
```

giving output like this:

```
Model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
Total parameters: 1,242M
Trainable parameters: 0.426M
Iteration 1: validation loss 2.443872, validation time 3.330629s
Iteration 10: training loss 2.356848, iterations/sec 2.640604, Tokens/sec 260.363581
Iteration 20: training loss 2.063395, iterations/sec 2.294999, Tokens/sec 232.483365
Iteration 30: training loss 1.63846, iterations/sec 2.279401, Tokens/sec 225.204788
Iteration 40: training loss 1.66366, iterations/sec 2.493669, Tokens/sec 218.196057
Iteration 50: training loss 1.470927, iterations/sec 2.301153, Tokens/sec 231.72614
Iteration 60: training loss 1.396581, iterations/sec 2.400012, Tokens/sec 230.401195
Iteration 70: training loss 1.587023, iterations/sec 2.422193, Tokens/sec 218.966258
Iteration 80: training loss 1.376895, iterations/sec 2.111973, Tokens/sec 216.477187
Iteration 90: training loss 1.245127, iterations/sec 2.383802, Tokens/sec 214.065436
Iteration 100: training loss 1.344523, iterations/sec 2.424746, Tokens/sec 223.076649
Iteration 100: validation loss 1.400582, validation time 3.489797s
Iteration 100: saved weights to /tmp/lora.safetensors
...
Iteration 910: training loss 1.181306, iterations/sec 2.355085, Tokens/sec 212.428628
Iteration 920: training loss 1.042286, iterations/sec 2.374377, Tokens/sec 222.479127
Iteration 930: training loss 0.920768, iterations/sec 2.475088, Tokens/sec 220.035347
Iteration 940: training loss 1.140762, iterations/sec 2.119886, Tokens/sec 227.039828
Iteration 950: training loss 1.068073, iterations/sec 2.523047, Tokens/sec 218.495903
Iteration 960: training loss 1.106662, iterations/sec 2.339293, Tokens/sec 221.063186
Iteration 970: training loss 0.833658, iterations/sec 2.474683, Tokens/sec 213.56517
Iteration 980: training loss 0.844026, iterations/sec 2.441064, Tokens/sec 210.663791
Iteration 990: training loss 0.903735, iterations/sec 2.253876, Tokens/sec 218.175162
Iteration 1000: training loss 0.872615, iterations/sec 2.343899, Tokens/sec 219.62336
Iteration 1000: validation loss 0.714194, validation time 3.470462s
Iteration 1000: saved weights to /tmp/lora-layers-4.safetensors
```

### Testing

You can test the LoRA adapated model against the `test` dataset using this command:

```
./mlx-run llm-tool lora test \ 
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --data Data/lora \
    --adapter /tmp/lora-layers-4.safetensors \
    --batch-size 1 --lora-layers 4 \
    --cache-size 1024
```

This will run all the items (100 in the example data we are using) in the test set and compute the loss:

```
Model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
Total parameters: 1,242M
Trainable parameters: 0.426M
Test loss 1.327623, ppl 3.772065
```

### Evaluate

Next you can evaluate your own prompts with the fine tuned LoRA adapters.  It is important to
follow the prompt example from the training data to match the format:

```
{"text": "table: 1-10015132-1\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: What school did player number 6 come from?\nA: SELECT School/Club Team FROM 1-10015132-1 WHERE No. = '6'"}
```

Given that format you might issue a command like this:

```
./mlx-run llm-tool lora eval \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --adapter /tmp/lora-layers-4.safetensors \
    --lora-layers 4 \
    --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

> Note: the prompt has newlines in it to match the format of the fine tuned prompts -- this may be easier to do with the command line than Xcode.

You might be treated to a response like this:

```
Model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
Total parameters: 1,242M
Trainable parameters: 0.426M
Starting generation ...
table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrence Ross' AND No. = 1
```

### Fusing

Once the adapter weights are trained you can produce new weights with the original achitecture that
have the adapter weights merged in:

```
./mlx-run llm-tool lora fuse \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --adapter /tmp/lora-layers-4.safetensors \
    --output mlx-community/mistral-lora
```

outputs:

```
Total parameters: 1,244M
Trainable parameters: 0.426M
Use with:
    llm-tool eval --model mlx-community/mistral-lora
```

As noted in the output these new weights can be used with the original model architecture.


[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.
[^qlora]: Refer to the paper [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
[^wikisql]: Refer to the [GitHub repo](https://github.com/salesforce/WikiSQL/tree/master) for more information about WikiSQL.
