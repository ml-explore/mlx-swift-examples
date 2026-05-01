//
//  ChatToolbarView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import SwiftUI

/// Toolbar content for the chat interface that displays error messages, download progress,
/// generation statistics, and model selection controls.
struct ChatToolbarView: ToolbarContent {
    /// View model containing the chat state and controls
    @Bindable var vm: ChatViewModel

    var body: some ToolbarContent {
        // Display error message if present
        if let errorMessage = vm.errorMessage {
            ToolbarItem {
                ErrorView(errorMessage: errorMessage)
            }
        }

        // Show download progress while bytes are flowing from the network
        if let progress = vm.modelDownloadProgress {
            ToolbarItem {
                DownloadProgressView(progress: progress)
            }
        } else if vm.isLoadingModel {
            // Otherwise, show a generic loading indicator while weights are
            // being initialized into memory.
            ToolbarItem {
                ProgressView()
                    .controlSize(.small)
                    .help("Loading model")
            }
            .sharedBackgroundVisibility(.hidden)
        }

        // Generation statistics indicator (only shown after a generation has completed)
        if let tokensPerSecond = vm.tokensPerSecond {
            ToolbarItem {
                GenerationInfoView(
                    tokensPerSecond: tokensPerSecond
                )
            }
            .sharedBackgroundVisibility(.hidden)
        }

        ToolbarSpacer(.fixed)

        // Button to clear chat history (only shown when there's something to clear)
        if vm.canClearChat {
            ToolbarItem {
                Button {
                    vm.clear([.chat, .meta])
                } label: {
                    Image(systemName: "trash")
                }
                .help("Clear chat")
                .disabled(vm.isGenerating)
            }
        }

        // Model selection picker
        ToolbarItem {
            Picker("Model", selection: $vm.selectedModel) {
                ForEach(MLXService.availableModels) { model in
                    Text(model.displayName)
                        .tag(model)
                }
            }
        }
    }
}
