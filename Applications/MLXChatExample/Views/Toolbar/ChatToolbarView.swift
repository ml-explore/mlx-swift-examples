//
//  ChatToolbarView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import SwiftUI

/// Toolbar view for the chat interface that displays error messages, download progress,
/// generation statistics, and model selection controls.
struct ChatToolbarView: View {
    /// View model containing the chat state and controls
    @Bindable var vm: ChatViewModel

    var body: some View {
        HStack(spacing: 12) {
            // Display error message if present
            if let errorMessage = vm.errorMessage {
                ErrorView(errorMessage: errorMessage)
            }

            // Show download progress for model loading
            if let progress = vm.modelDownloadProgress, !progress.isFinished {
                DownloadProgressView(progress: progress)
            }

            // Button to clear chat history, displays generation statistics
            Button {
                vm.clear([.chat, .meta])
            } label: {
                GenerationInfoView(
                    tokensPerSecond: vm.tokensPerSecond
                )
            }

            // Model selection picker
            Picker("Model", selection: $vm.selectedModel) {
                ForEach(MLXService.availableModels) { model in
                    Text(model.displayName)
                        .tag(model)
                }
            }
            .frame(width: 120)

            // Assistant display pciker
            Picker("Display Mode", selection: $vm.displayMode) {
                ForEach(MessageDisplayMode.allCases) { mode in
                    Text(mode.rawValue.capitalized)
                        .tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 160)
        }
    }
}
