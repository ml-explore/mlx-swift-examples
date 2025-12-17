// Copyright © 2025 Apple Inc.

import MLXLMCommon
import SwiftUI

struct ContentView: View {

    /// provided by the application
    let loader: ModelLoader

    /// once loaded this will hold the chat session
    @State var session: ChatModel?
    @State var error: String?

    /// prompt for the LLM (text field)
    @State var prompt = ""

    @FocusState var promptFocused

    var body: some View {
        VStack {
            if let error {
                Text("Error: \(error)")

            } else if !loader.isLoaded {
                ProgressView("Loading", value: loader.progress, total: 1)

            } else if let session {
                // show the chat messages
                ScrollView(.vertical) {
                    ForEach(session.messages.enumerated(), id: \.offset) { _, message in
                        let bold = message.role == .user

                        HStack {
                            Text(message.content)
                                .bold(bold)
                            Spacer()
                        }
                        .padding(.bottom, 4)
                    }

                    Spacer()

                    if session.isBusy {
                        // a stop button -- cmd-. to interrupt
                        HStack {
                            Button("Stop", action: { session.cancel() })
                                .keyboardShortcut(".")
                            Spacer()
                        }
                    } else {
                        // message from the user -> LLM
                        TextField("Prompt", text: $prompt)
                            .onSubmit(respond)
                            .focused($promptFocused)
                            .onAppear {
                                promptFocused = true
                            }
                    }
                }
                .defaultScrollAnchor(.bottom)
            }
        }
        .padding()
        .task {
            do {
                let model = try await loader.model()
                self.session = ChatModel(model: model)
            } catch {
                self.error = error.localizedDescription
            }
        }
        .onDisappear {
            self.session?.cancel()
        }
    }

    private func respond() {
        session?.respond(prompt)
        prompt = ""
    }
}
