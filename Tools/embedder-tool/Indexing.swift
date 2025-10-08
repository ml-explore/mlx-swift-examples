import Foundation

struct Document {
    let path: String
    let contents: String
}

struct IndexEntry: Codable {
    let path: String
    let embedding: [Float]
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

    func load() throws -> [Document] {
        let fileManager = FileManager.default
        var documents: [Document] = []

        if recursive {
            guard let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ) else {
                return []
            }

            for case let fileURL as URL in enumerator {
                guard try shouldInclude(url: fileURL) else { continue }
                if let document = try readDocument(at: fileURL) {
                    documents.append(document)
                }
                if reachedLimit(current: documents.count) { break }
            }
        } else {
            let items = try fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )

            for fileURL in items {
                guard try shouldInclude(url: fileURL) else { continue }
                if let document = try readDocument(at: fileURL) {
                    documents.append(document)
                }
                if reachedLimit(current: documents.count) { break }
            }
        }

        return documents
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
        do {
            let contents = try String(contentsOf: url)
            return Document(path: relativePath(for: url), contents: contents)
        } catch {
            return nil
        }
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
}
