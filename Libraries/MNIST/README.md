#  MNIST

This is a port of the MNIST training code from the [Python MLX example](https://github.com/ml-explore/mlx-examples/blob/main/mnist). This example uses a [LeNet](https://en.wikipedia.org/wiki/LeNet) instead of an MLP.

It provides code to:

- Download the MNIST test/train data
- Build the LeNet
- Some functions to shuffle and batch the data

See [mnist-tool](../../Tools/mnist-tool) for an example of how to run this. The training loop also lives there.
