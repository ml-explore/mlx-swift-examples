import ArgumentParser
import Foundation

struct ReplCommand: EmbedderCommand {
    static let configuration = CommandConfiguration(
        commandName: "repl",
        abstract: "Interactively search embeddings built from a local directory"
    )

    @OptionGroup var model: ModelArguments
    @OptionGroup var corpus: CorpusArguments
    @OptionGroup var pooling: PoolingArguments
    @OptionGroup var memory: MemoryArguments

    @Option(
        name: .shortAndLong,
        help: "Number of matches to display for each query (default: 5)"
    )
    var top: Int = 5

    @Option(
        name: .long,
        help: "Number of documents to embed per batch (default: 8)"
    )
    var batchSize: Int = 8

    @Flag(
        name: .long,
        help: "Print timing information for each query"
    )
    var showTiming = false

    mutating func run(runtime: EmbedderRuntime) async throws {
        let loadResult = try loadCorpus()

        guard !loadResult.documents.isEmpty else {
            print("No documents found in \(corpus.directoryURL.path)")
            return
        }

        print(
            "Embedding \(loadResult.documents.count) document(s) from \(corpus.directoryURL.path)")
        let index = try await embed(
            documents: loadResult.documents, runtime: runtime, batchSize: batchSize)

        guard !index.isEmpty else {
            print("No embeddings were generated for the selected documents")
            return
        }

        let stats = EmbeddingStats(
            documentCount: index.count,
            embeddingDimension: index.first?.embedding.count
        )

        printStartupStats(stats: stats)
        await runLoop(runtime: runtime, entries: index, stats: stats)
    }

    private func loadCorpus() throws -> CorpusLoader.LoadResult {
        let loader = CorpusLoader(
            root: corpus.directoryURL,
            extensions: corpus.normalizedExtensions,
            recursive: corpus.recursive,
            limit: corpus.limit
        )
        let result = try loader.load()
        if !result.failures.isEmpty {
            reportCorpusFailures(result.failures)
        }
        return result
    }

    private func embed(documents: [Document], runtime: EmbedderRuntime, batchSize: Int) async throws
        -> [IndexEntry]
    {
        let batchSize = max(1, min(batchSize, documents.count))
        var accumulatedEntries: [IndexEntry] = []
        var skippedDocuments: [String] = []
        var fallbackMessages: Set<String> = []
        var processed = 0

        while processed < documents.count {
            let upperBound = min(processed + batchSize, documents.count)
            let batch = Array(documents[processed ..< upperBound])
            let result = try await runtime.embed(texts: batch.map { $0.contents })

            let entries = result.embeddings.compactMap { embedding -> IndexEntry? in
                guard batch.indices.contains(embedding.index) else { return nil }
                let document = batch[embedding.index]
                let vector =
                    runtime.normalize
                    ? VectorOperations.normalize(embedding.vector)
                    : VectorOperations.sanitize(embedding.vector)
                return IndexEntry(path: document.path, embedding: vector)
            }

            accumulatedEntries.append(contentsOf: entries)

            skippedDocuments.append(
                contentsOf: result.skippedIndices.compactMap { index -> String? in
                    guard batch.indices.contains(index) else { return nil }
                    return batch[index].path
                })

            if let message = result.fallbackDescription {
                fallbackMessages.insert(message)
            }

            processed = upperBound
        }

        if !skippedDocuments.isEmpty {
            reportSkippedDocuments(skippedDocuments)
        }
        for message in fallbackMessages {
            reportPoolingFallback(message)
        }

        return accumulatedEntries
    }

    private func runLoop(runtime: EmbedderRuntime, entries: [IndexEntry], stats: EmbeddingStats)
        async
    {
        print(
            "Enter a query to search, or /help for commands. Press return on an empty line to exit."
        )

        while true {
            prompt()
            guard let line = readLine() else {
                print("\nEnd of input; exiting")
                break
            }

            let query = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if query.isEmpty {
                print("Exiting")
                break
            }

            switch handleCommand(query, stats: stats) {
            case .notCommand:
                break
            case .handled:
                continue
            case .quit:
                print("Exiting")
                return
            }

            await handleQuery(query, runtime: runtime, entries: entries)
        }
    }

    private func handleQuery(_ query: String, runtime: EmbedderRuntime, entries: [IndexEntry]) async
    {
        let start = showTiming ? DispatchTime.now() : nil
        var shouldReportTime = false
        defer {
            if showTiming, shouldReportTime, let start {
                let elapsed = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
                let milliseconds = Double(elapsed) / 1_000_000.0
                print(String(format: "query time: %.2f ms", milliseconds))
            }
        }

        guard var queryVector = await embedQuery(query, runtime: runtime) else {
            return
        }
        shouldReportTime = true

        guard !queryVector.isEmpty else {
            shouldReportTime = false
            print("Query produced no tokens; please try different text")
            return
        }

        if VectorOperations.hasNonFiniteValues(queryVector) {
            shouldReportTime = false
            writeDiagnostic(
                "Query vector contains non-finite values; skipping search", kind: .error)
            return
        }

        queryVector = VectorOperations.sanitize(queryVector)

        if runtime.normalize {
            queryVector = VectorOperations.normalize(queryVector)
        }

        let (ranked, mismatched) = rank(entries: entries, query: queryVector)

        if !mismatched.isEmpty {
            writeDiagnostic(
                dimensionMismatchMessage(for: mismatched, expected: queryVector.count),
                kind: .warning)
        }

        guard !ranked.isEmpty else {
            shouldReportTime = false
            print("No comparable embeddings in memory")
            return
        }

        let limit = max(0, min(top, ranked.count))
        for (entry, score) in ranked.prefix(limit) {
            print(String(format: "%@\t%.4f", entry.path, score))
        }
    }

    private func embedQuery(_ query: String, runtime: EmbedderRuntime) async -> [Float]? {
        do {
            let result = try await runtime.embed(texts: [query])
            if let message = result.fallbackDescription {
                writeDiagnostic(message, kind: .warning)
            }
            guard let embedding = result.embeddings.first(where: { $0.index == 0 }) else {
                return []
            }
            return embedding.vector
        } catch {
            writeDiagnostic("Failed to embed query: \(error.localizedDescription)", kind: .error)
            return nil
        }
    }

    private func rank(entries: [IndexEntry], query: [Float]) -> (
        [(IndexEntry, Float)], [(path: String, dimension: Int)]
    ) {
        var mismatched: [(String, Int)] = []

        let ranked = entries.compactMap { entry -> (IndexEntry, Float)? in
            let dimension = entry.embedding.count
            guard dimension == query.count else {
                mismatched.append((entry.path, dimension))
                return nil
            }

            guard !VectorOperations.hasNonFiniteValues(entry.embedding) else {
                return (entry, 0)
            }

            let score = VectorOperations.dotProduct(entry.embedding, query)
            return (entry, score)
        }
        .sorted { $0.1 > $1.1 }

        return (ranked, mismatched)
    }

    private func dimensionMismatchMessage(
        for mismatched: [(path: String, dimension: Int)], expected: Int
    ) -> String {
        var message =
            "Skipped \(mismatched.count) index entry(s) with dimension mismatch (expected \(expected))"
        if !mismatched.isEmpty {
            let preview = mismatched.prefix(5).map { "\($0.path) (\($0.dimension))" }.joined(
                separator: ", ")
            message += ": \(preview)"
            if mismatched.count > 5 {
                message += ", ..."
            }
        }
        return message
    }

    private func prompt() {
        if let data = "query> ".data(using: .utf8) {
            FileHandle.standardOutput.write(data)
        }
    }

    private func handleCommand(_ input: String, stats: EmbeddingStats) -> CommandAction {
        guard input.hasPrefix("/") else { return .notCommand }
        let command = input.lowercased()
        switch command {
        case "/help":
            printHelp()
            return .handled
        case "/stats":
            printStats(stats: stats)
            return .handled
        case "/quit", "/exit":
            return .quit
        default:
            print("Unknown command \(input). Type /help for a list of commands.")
            return .handled
        }
    }

    private func printHelp() {
        print(
            """
            Available commands:
              /help   Show this message
              /stats  Display embedding statistics
              /quit   Exit the REPL
            """)
    }

    private func printStats(stats: EmbeddingStats) {
        var lines: [String] = []
        lines.append("Documents indexed: \(stats.documentCount)")
        if let dimension = stats.embeddingDimension {
            lines.append("Embedding dimension: \(dimension)")
        }
        print(lines.joined(separator: "\n"))
    }

    private func printStartupStats(stats: EmbeddingStats) {
        print("Loaded embeddings for \(stats.documentCount) document(s)")
        if let dimension = stats.embeddingDimension {
            print("Embedding dimension: \(dimension)")
        }
    }

    private func reportSkippedDocuments(_ paths: [String]) {
        var message = "Skipped \(paths.count) document(s) that produced no tokens"
        if !paths.isEmpty {
            let preview = paths.prefix(5).joined(separator: ", ")
            message += ": \(preview)"
            if paths.count > 5 {
                message += ", ..."
            }
        }
        writeDiagnostic(message, kind: .warning)
    }

    private func reportPoolingFallback(_ message: String) {
        writeDiagnostic(message, kind: .warning)
    }

    private func reportCorpusFailures(_ failures: [(url: URL, error: CorpusLoader.ReadError)]) {
        var message = "Skipped \(failures.count) file(s) while reading corpus"
        if !failures.isEmpty {
            let preview = failures.prefix(5).map { failure in
                let description = failure.error.errorDescription ?? "unreadable"
                return "\(failure.url.lastPathComponent) (\(description))"
            }.joined(separator: ", ")
            message += ": \(preview)"
            if failures.count > 5 {
                message += ", ..."
            }
        }
        writeDiagnostic(message, kind: .warning)
    }
}

private struct EmbeddingStats {
    let documentCount: Int
    let embeddingDimension: Int?
}

private enum CommandAction {
    case notCommand
    case handled
    case quit
}
