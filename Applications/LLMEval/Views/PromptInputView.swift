// Copyright © 2025 Apple Inc.

import SwiftUI

struct PromptInputView: View {
    @Bindable var llm: LLMEvaluator
    @Binding var isPromptExpanded: Bool
    @Binding var showingPresetPrompts: Bool

    let onGenerate: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Prompt header with expand/collapse chevron
            HStack {
                Text("Prompt")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isPromptExpanded.toggle()
                    }
                } label: {
                    Image(systemName: isPromptExpanded ? "chevron.down" : "chevron.up")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help(isPromptExpanded ? "Collapse prompt area" : "Expand prompt area")
            }

            // Prompt text field with dynamic sizing
            TextField("Enter your prompt...", text: $llm.prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(isPromptExpanded ? 15 ... 50 : 1 ... 3)
                .frame(height: isPromptExpanded ? 400 : nil)
                .onSubmit(onGenerate)
                .disabled(llm.running || llm.isLoading)

            // Action buttons
            HStack(spacing: 12) {
                Button {
                    showingPresetPrompts = true
                } label: {
                    Label("Example Prompts", systemImage: "list.bullet")
                }
                .disabled(llm.running || llm.isLoading)

                Spacer()

                Button {
                    if llm.running {
                        onCancel()
                    } else {
                        onGenerate()
                    }
                } label: {
                    Label(
                        llm.running ? "Stop" : "Generate",
                        systemImage: llm.running ? "stop.circle" : "play.fill"
                    )
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled((llm.prompt.isEmpty && !llm.running) || llm.isLoading)
            }
        }
        .padding(.vertical, 8)
    }
}
