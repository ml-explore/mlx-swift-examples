import ArgumentParser
import Foundation

struct DemoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "demo",
        abstract: "Run a brief demonstration using sample repository documentation"
    )

    @Flag(name: .long, help: "Keep the generated demo index file instead of removing it")
    var keepIndex = false

    func run() async throws {
        print("Embedder Tool Demo")
        print(String(repeating: "=", count: 20))
        print()

        let indexURL = try makeTemporaryIndexURL()
        defer {
            if !keepIndex {
                try? FileManager.default.removeItem(at: indexURL)
            }
        }

        try await buildIndex(at: indexURL)
        print()
        try await runSampleQueries(using: indexURL)
        print()
        print("Try it yourself:")
        print("  embedder-tool index --directory <your-docs>")
        print("  embedder-tool search --index <index-file> --query \"your query\"")
        if keepIndex {
            print()
            print("Demo index saved at \(indexURL.path)")
        }
    }

    private func makeTemporaryIndexURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
        return directory.appendingPathComponent("embedder-demo-\(UUID().uuidString).json")
    }

    private func buildIndex(at url: URL) async throws {
        print("Indexing sample documentation from Libraries/")

        let arguments = [
            "--output", url.path,
            "--directory", "Libraries",
            "--extensions", "md",
            "--recursive",
            "--limit", "8",
            "--batch-size", "4",
            "--normalize"
        ]

        let indexCommand = try IndexCommand.parse(arguments)
        try await indexCommand.run()
    }

    private func runSampleQueries(using indexURL: URL) async throws {
        let queries = [
            "How do I use embedding models?",
            "Training language models",
            "Vision language models"
        ]

        for query in queries {
            print("Query: \"\(query)\"")
            let arguments = [
                "--index", indexURL.path,
                "--query", query,
                "--top", "2",
                "--normalize"
            ]
            let searchCommand = try SearchCommand.parse(arguments)
            try await searchCommand.run()
            print()
        }
    }
}
