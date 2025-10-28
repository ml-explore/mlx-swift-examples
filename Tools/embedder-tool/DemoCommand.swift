import ArgumentParser
import Foundation

struct DemoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "demo",
        abstract: "Run a demo using sample repository documentation"
    )

    @OptionGroup var memory: MemoryArguments

    @Flag(name: .long, help: "Keep the generated demo index file instead of removing it")
    var keepIndex = false

    @Argument(help: "Optional queries to run after indexing. Defaults to three sample queries.")
    var queries: [String] = []

    mutating func run() async throws {
        var memory = self.memory
        memory.start()
        defer {
            memory.reportMemoryStatistics()
            self.memory = memory
        }

        print("Embedder Tool Demo")

        let indexURL = try makeTemporaryIndexURL()
        defer {
            if !keepIndex {
                do {
                    try FileManager.default.removeItem(at: indexURL)
                } catch {
                    if FileManager.default.fileExists(atPath: indexURL.path) {
                        let message =
                            "Failed to remove temporary index file at \(indexURL.path): \(error.localizedDescription). Please remove it manually."
                        writeDiagnostic(message, kind: .warning)
                    }
                }
            }
        }

        try await buildIndex(at: indexURL)
        let queriesToRun = queries.isEmpty ? defaultQueries : queries
        try await runSampleQueries(using: indexURL, queries: queriesToRun)
    }

    private func makeTemporaryIndexURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
        return directory.appendingPathComponent("embedder-demo-\(UUID().uuidString).json")
    }

    private func buildIndex(at url: URL) async throws {
        var indexCommand = IndexCommand()
        indexCommand.corpus.directory = URL(fileURLWithPath: "Libraries")
        indexCommand.corpus.extensions = ["md"]
        indexCommand.corpus.recursive = true
        indexCommand.corpus.limit = 8
        indexCommand.output = url
        indexCommand.batchSize = 4
        indexCommand.pooling.normalize = true

        try await indexCommand.run()
    }

    private func runSampleQueries(using indexURL: URL, queries: [String]) async throws {
        for query in queries {
            var searchCommand = SearchCommand()
            searchCommand.index = indexURL
            searchCommand.query = query
            searchCommand.top = 2
            searchCommand.pooling.normalize = true

            try await searchCommand.run()
        }
    }

    private var defaultQueries: [String] {
        [
            "How do I use embedding models?",
            "Training language models",
            "Vision language models",
        ]
    }
}
