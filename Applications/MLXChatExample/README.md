# MLX Chat Example

A lightweight chat application demonstrating MLX integration for iOS and macOS. Built with SwiftUI, this example project shows how to implement both Large Language Models (LLMs) and Vision Language Models (VLMs) using MLX.

<img alt="MLX Chat Example Screenshot" src="https://github.com/user-attachments/assets/9a20c081-61c2-4b0a-88df-f54500464d77" />

## Features

- 🤖 LLM and VLM support with real-time text generation
- 📱 Cross-platform (iOS and macOS)
- 🖼️ Image and video input for vision models
- 💾 Efficient model caching and memory management
- ⚡️ Async/await based generation with cancellation support
- 🎨 Modern SwiftUI interface
- 📝 Comprehensive documentation and comments

## Requirements

- iOS 17.0+ / macOS 14.0+
- Xcode 15.0+
- Swift 5.9+

## Dependencies

- MLX: Core machine learning operations
- MLXLMCommon: Common language model utilities
- MLXLLM: Large language model support
- MLXVLM: Vision-language model support

## Project Structure

```
MLXChatExample/
├── Views/                # SwiftUI views
├── Models/               # Data models
├── ViewModels/           # Business logic
├── Services/             # MLX integration
└── Support/              # Utilities
```

## Technical Overview

The project follows MVVM architecture with clear separation between UI and business logic. The core functionality is split into two main components:

### MLXService

Core service handling all model operations:
- Model loading and caching with memory management
- Async text generation with streaming support
- GPU memory optimization
- Model state management
- Handles both LLM and VLM model types

### ChatViewModel

Business logic coordinator:
- Manages chat state and message history
- Handles generation lifecycle and cancellation
- Coordinates media attachments for vision models
- Performance metrics and error handling
- Provides clean interface between UI and ML service

### Architecture Highlights

- Complete separation of UI and business logic
- SwiftUI views with async/await integration
- Modular design for easy extension

### Documentation

The codebase is thoroughly documented with:
- Detailed class and method documentation
- Clear inline comments explaining complex logic
- DocC documentation format

### Markdown Support

This sample app renders markdown content using SwiftUI's native `Text` view by passing the content as a `LocalizedStringKey`:

```swift
Text(LocalizedStringKey(message.content))
```

#### Limitations and Alternatives

The default SwiftUI markdown rendering only supports standard markdown syntax. It does not support advanced features like tables and task lists that are available in GitHub Flavored Markdown (GFM).

For more comprehensive markdown support:

- **GitHub Flavored Markdown**: Consider using the [swift-markdown-ui](https://github.com/gonzalezreal/swift-markdown-ui) library. However, be aware that this library currently has an [unresolved issue with text selection](https://github.com/gonzalezreal/swift-markdown-ui/issues/264), which is why it wasn't used in this example.

- **Enhanced Text Selection**: If you're satisfied with standard markdown but want better text selection capabilities on iOS (instead of only being able to select and copy entire content block), consider combining:
  - [SelectableText](https://github.com/kevinhermawan/SelectableText) for improved selection functionality
  - [MarkdownToAttributedString](https://github.com/madebywindmill/MarkdownToAttributedString) for markdown formatting

> More discussion on this can be found on [issue #297](https://github.com/ml-explore/mlx-swift-examples/issues/297)

## Getting Started

1. Clone the repository
2. Install dependencies
3. Open in Xcode
4. Build and run

## Contributing

This is an example project demonstrating MLX capabilities. Feel free to use it as a reference for your own projects.

## Acknowledgments

- MLX team for the core framework
