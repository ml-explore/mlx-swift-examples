// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation

struct CorpusArguments: ParsableArguments {

    @Option(name: .shortAndLong, help: "Directory containing documents to index.")
    var directory: URL = URL(
        fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)

    @Option(
        name: [.customShort("e"), .long], parsing: .upToNextOption,
        help: "File extensions to include (without dots).")
    var extensions: [String] = ["txt", "md"]

    @Flag(name: .shortAndLong, help: "Recursively scan subdirectories.")
    var recursive = false

    @Option(name: .long, help: "Limit the number of documents to load.")
    var limit: Int?

    var normalizedExtensions: [String] {
        extensions.map { value in
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            let noDot = trimmed.hasPrefix(".") ? String(trimmed.dropFirst()) : trimmed
            return noDot.lowercased()
        }
    }

    var directoryURL: URL { directory.standardizedFileURL }
}
