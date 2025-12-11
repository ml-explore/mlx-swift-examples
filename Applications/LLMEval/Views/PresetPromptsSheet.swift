// Copyright © 2025 Apple Inc.

import SwiftUI

struct PresetPromptsSheet: View {
    @Binding var isPresented: Bool
    let onSelect: (PresetPrompt) -> Void

    var body: some View {
        NavigationStack {
            List {
                ForEach(PresetPrompts.all) { preset in
                    Button {
                        onSelect(preset)
                        isPresented = false
                    } label: {
                        HStack(alignment: .center, spacing: 12) {
                            // Show the actual prompt (first 2 lines)
                            // Clean up whitespace for better preview
                            let cleanedPrompt = preset.prompt
                                .trimmingCharacters(in: .whitespacesAndNewlines)
                                .replacingOccurrences(
                                    of: #"\n\n+"#, with: " ", options: .regularExpression
                                )
                                .replacingOccurrences(
                                    of: #"\s+"#, with: " ", options: .regularExpression)

                            Text(cleanedPrompt)
                                .multilineTextAlignment(.leading)
                                .lineLimit(2)
                                .font(.body)
                                .foregroundStyle(.primary)
                                .frame(maxWidth: .infinity, alignment: .leading)

                            // Show indicators if present
                            if preset.enableThinking || preset.enableTools || preset.isLongPrompt {
                                HStack(spacing: 6) {
                                    if preset.enableThinking {
                                        BadgeView(icon: "brain", text: "Thinking", color: .purple)
                                    }
                                    if preset.enableTools {
                                        BadgeView(icon: "hammer.fill", text: "Tools", color: .blue)
                                    }
                                    if preset.isLongPrompt {
                                        BadgeView(
                                            icon: "doc.text.fill", text: "Long", color: .orange)
                                    }
                                }
                            }
                        }
                        .padding(.vertical, 8)
                        #if os(macOS)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                        #endif
                    }
                    .buttonStyle(.plain)
                    #if os(macOS)
                        .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                    #endif
                }
            }
            #if os(macOS)
                .listStyle(.inset)
            #endif
            .navigationTitle("Example Prompts")
            #if !os(macOS)
                .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        isPresented = false
                    }
                }
            }
        }
        #if os(macOS)
            .frame(minWidth: 600, minHeight: 500)
        #endif
    }
}

// Badge component
private struct BadgeView: View {
    let icon: String
    let text: String
    let color: Color

    var body: some View {
        Label(text, systemImage: icon)
            .font(.caption2)
            .fontWeight(.medium)
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.gradient, in: Capsule())
    }
}
