import MLXLMCommon

let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")

let prompt = "What are three things to see in Paris?"

// MARK: - one-shot print out full response
print(
    """
    ================
    \(prompt)

    """)
print(try await ChatSession(model).respond(to: prompt))

// MARK: - one-shot streaming output
print(
    """
    ================
    \(prompt)

    """)
for try await item in ChatSession(model).streamResponse(to: prompt) {
    print(item, terminator: "")
}
print()

// MARK: - conversation with follow-on questions
let session = ChatSession(model)

let questions = [
    "What are two things to see in San Francisco?",
    "How about a great place to eat?",
    "What city are we talking about?  I forgot!",
]

for question in questions {
    print(
        """
        ================
        \(question)

        """)

    for try await item in session.streamResponse(to: question) {
        print(item, terminator: "")
    }
    print()
}
