// Copyright © 2025 Apple Inc.

import Foundation

struct Document {
    let path: String
    let contents: String
}

struct IndexEntry: Codable {
    let path: String
    let embedding: [Float]

    init(path: String, embedding: [Float]) {
        self.path = path
        self.embedding = embedding
    }
}

struct CorpusLoader {
    private let root: URL
    private let extensions: Set<String>
    private let recursive: Bool
    private let limit: Int?

    init(root: URL, extensions: [String], recursive: Bool, limit: Int?) {
        self.root = root.standardizedFileURL
        self.extensions = Set(extensions.map { $0.lowercased() })
        self.recursive = recursive
        self.limit = limit
    }

    struct LoadResult {
        let documents: [Document]
        let failures: [(url: URL, error: ReadError)]

        var successCount: Int { documents.count }
        var failureCount: Int { failures.count }
        var totalCount: Int { successCount + failureCount }
    }

    enum ReadError: LocalizedError {
        case binary
        case decoding
        case unreadable(String)

        var errorDescription: String? {
            switch self {
            case .binary:
                return "binary content"
            case .decoding:
                return "invalid text encoding"
            case .unreadable(let message):
                return message
            }
        }
    }

    func load() throws -> LoadResult {
        let fileManager = FileManager.default
        var documents: [Document] = []
        var failures: [(URL, ReadError)] = []

        let fileURLs: [URL]
        if recursive {
            guard
                let enumerator = fileManager.enumerator(
                    at: root,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles]
                )
            else {
                return LoadResult(documents: [], failures: [])
            }
            fileURLs = enumerator.allObjects as? [URL] ?? []
        } else {
            fileURLs = try fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        }

        for fileURL in fileURLs {
            guard try shouldInclude(url: fileURL) else { continue }
            do {
                if let document = try readDocument(at: fileURL) {
                    documents.append(document)
                }
            } catch let readError as ReadError {
                failures.append((fileURL, readError))
            } catch {
                failures.append((fileURL, .unreadable(error.localizedDescription)))
            }
            if reachedLimit(current: documents.count) { break }
        }

        return LoadResult(documents: documents, failures: failures)
    }

    private func shouldInclude(url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true else { return false }
        guard extensions.isEmpty || extensions.contains(url.pathExtension.lowercased()) else {
            return false
        }
        return true
    }

    private func readDocument(at url: URL) throws -> Document? {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ReadError.unreadable(error.localizedDescription)
        }

        if isLikelyBinary(data) {
            throw ReadError.binary
        }

        guard let contents = String(data: data, encoding: .utf8) else {
            throw ReadError.decoding
        }

        return Document(path: relativePath(for: url), contents: contents)
    }

    private func relativePath(for url: URL) -> String {
        let path = url.standardizedFileURL.path
        let base = root.path
        guard path.hasPrefix(base) else { return path }
        let suffixIndex = path.index(path.startIndex, offsetBy: base.count)
        let remainder = path[suffixIndex...]
        if remainder.hasPrefix("/") {
            return String(remainder.dropFirst())
        }
        return String(remainder)
    }

    private func reachedLimit(current: Int) -> Bool {
        guard let limit else { return false }
        return current >= limit
    }

    private func isLikelyBinary(_ data: Data) -> Bool {
        let sample = data.prefix(4096)
        return sample.contains { $0 == 0 }
    }
}
