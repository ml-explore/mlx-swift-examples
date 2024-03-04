#  MNISTTrainer

This is an example of model training that works on both macOS and iOS.
The example will download the MNIST training data, create a LeNet, and train
it. It will show the epoch time and test accuracy as it trains.

You will need to set the Team on the MNISTTrainer target in order to build and
run on iOS.

Some notes about the setup:

- This will download test data over the network so MNISTTrainer -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- The website it connects to uses http rather than https so it has a "App Transport Security Settings" in the Info.plist
