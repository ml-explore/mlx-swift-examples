#  mnist-tool

See the [MNIST README.md](../../Libraries/MNIST/README.md).

### Building

`mnist-tool` has no dependencies outside of the package dependencies
represented in xcode.

When you run the tool it will download the test/train datasets and
store them in a specified directory (see run arguments -- default is /tmp).

Simply build the project in xcode.

### Running (Xcode)

To run this in Xcode simply press cmd-opt-r to set the scheme arguments.  For example:

```
--data /tmp
```

Then cmd-r to run.

### Running (CommandLine)

Use the `mlx-run` script to run the command line tools:

```
./mlx-run mnist-tool --data /tmp
```

By default this will find and run the tools built in _Release_ configuration.  Specify `--debug`
to find and run the tool built in _Debug_ configuration.

See also:

- [MLX troubleshooting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/troubleshooting)
