#  MNISTTrainer

This is an example showing how to do model training on both macOS and iOS.
This will download the MNIST training data, create a new models and train
it.  It will show the timing per epoch and the test accuracy as it trains.

You will need to set the Team on the MNISTTrainer target in order to build and
run on iOS.

Some notes about the setup:

- this will download test data over the network so MNISTTrainer -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- the website it connects to uses http rather than https so it has a "App Transport Security Settings" in the Info.plist
