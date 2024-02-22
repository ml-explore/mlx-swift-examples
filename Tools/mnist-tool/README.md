#  mnist-tool

See other README:

- [MNIST](../../Libraries/MNIST/README.md)

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

`mnist-tool` can also be run from the command line if built from Xcode, but 
the `DYLD_FRAMEWORK_PATH` must be set so that the frameworks and bundles can be found:

- [MLX troubleshooting](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/troubleshooting)

```
DYLD_FRAMEWORK_PATH=~/Library/Developer/Xcode/DerivedData/mlx-examples-swift-ceuohnhzsownvsbbleukxoksddja/Build/Products/Debug ~/Library/Developer/Xcode/DerivedData/mlx-examples-swift-ceuohnhzsownvsbbleukxoksddja/Build/Products/Debug/mnist-tool --data /tmp
```
