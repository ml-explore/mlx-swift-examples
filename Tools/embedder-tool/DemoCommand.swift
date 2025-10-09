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

        var indexCommand = IndexCommand()
        indexCommand.output = url.path
        indexCommand.batchSize = 4

        var corpus = CorpusArguments()
        corpus.directory = "Libraries"
        corpus.extensions = ["md"]
        corpus.recursive = true
        corpus.limit = 8
        indexCommand.corpus = corpus

        var pooling = PoolingArguments()
        pooling.normalize = true
        indexCommand.pooling = pooling

        indexCommand.model = ModelArguments()

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
            var searchCommand = SearchCommand()
            searchCommand.model = ModelArguments()
            var pooling = PoolingArguments()
            pooling.normalize = true
            searchCommand.pooling = pooling
            searchCommand.index = indexURL.path
            searchCommand.query = query
            searchCommand.top = 2
            try await searchCommand.run()
            print()
        }
    }
}
