#  LLMBasic

A minimal example of:

- loading a model, including downloading weights
- setting up a ChatSession
- a simple UI for a back and forth session with the model

The `ChatModel` has a few parameters at the top if you want to try a different model or
system prompt.

The goal of this example is to be a **minimal** application that loads and interacts with
an LLM.

See `LLMEval` and `MLXChatExample` for more full featured applications.

As always, you must set the Team on the LLMBasic target.

Some notes about the setup:

- this downloads models from hugging face so LLMBasic -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- LLM models are large so this uses the Increased Memory Limit entitlement on iOS to allow ... increased memory limits for devices that have more memory
- `Memory.cacheLimit = 20 * 1024 * 1024` is used to limit the buffer cache size
