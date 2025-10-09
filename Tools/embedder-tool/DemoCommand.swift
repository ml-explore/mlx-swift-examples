import ArgumentParser
import Foundation

struct DemoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "demo",
        abstract: "Run a demo using sample repository documentation"
    )

    @Flag(name: .long, help: "Keep the generated demo index file instead of removing it")
    var keepIndex = false

    func run() async throws {
        print("Embedder Tool Demo")

        let indexURL = try makeTemporaryIndexURL()
        defer {
            if !keepIndex {
                do {
                    try FileManager.default.removeItem(at: indexURL)
                } catch {
                    if FileManager.default.fileExists(atPath: indexURL.path) {
                        print("Failed to remove temporary index file: \(error.localizedDescription)")
                    }
                }
            }
        }

        try await buildIndex(at: indexURL)
        try await runSampleQueries(using: indexURL)
    }

    private func makeTemporaryIndexURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
        return directory.appendingPathComponent("embedder-demo-\(UUID().uuidString).json")
    }

    private func buildIndex(at url: URL) async throws {
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
        }
    }
}
