// Copyright 2024 Apple Inc.

import Foundation

enum LoRADataError: Error {
    case fileNotFound(URL, String)
    case invalidJSON(String, Int)
}

/// Load a LoRA data file.
///
/// Given a directory and a base name, e.g. `train`, this will load a `.jsonl` or `.txt` file
/// if possible.
public func loadLoRAData(directory: URL, name: String) throws -> [String] {
    let extensions = ["jsonl", "txt"]

    for ext in extensions {
        let url = directory.appending(component: "\(name).\(ext)")
        if FileManager.default.fileExists(atPath: url.path()) {
            return try loadLoRAData(url: url)
        }
    }

    throw LoRADataError.fileNotFound(directory, name)
}

/// Load a .txt or .jsonl file and return the contents
public func loadLoRAData(url: URL) throws -> [String] {
    switch url.pathExtension {
    case "jsonl":
        return try loadJSONL(url: url)

    case "txt":
        return try loadLines(url: url)

    default:
        fatalError("Unable to load data file, unknown type: \(url)")

    }
}

func loadJSONL(url: URL) throws -> [String] {
    let content = try String(contentsOf: url).components(separatedBy: .newlines)
    var results = [String]()
    var decoder: ((Data) throws -> String)?
    
    for (index, line) in content.enumerated() {
        guard !line.trimmingCharacters(in: .whitespaces).isEmpty else {
            continue
        }
        
        if decoder == nil {
            decoder = try determineDecoder(line: line)
        }
        
        if let decodeFunction = decoder {
            let data = Data(line.utf8)
            let decodedLine = try decodeFunction(data)
            results.append(decodedLine)
        } else {
            throw LoRADataError.invalidJSON(line, index + 1)
        }
    }
    
    return results
}

private func determineDecoder(line: String) throws -> (Data) throws -> String {
    let data = Data(line.utf8)
    let decoder = JSONDecoder()

    if let _ = try? decoder.decode(ChatStructure.self, from: data) {
        return { data in
            let chatStruct = try JSONDecoder().decode(ChatStructure.self, from: data)
            return chatStruct.messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n")
        }
    }
    
    if let _ = try? decoder.decode(ToolStructure.self, from: data) {
        return { _ in line }
    }
    
    if let _ = try? decoder.decode(TextStructure.self, from: data) {
        return { data in
            let textStruct = try JSONDecoder().decode(TextStructure.self, from: data)
            return textStruct.text
        }
    }
    
    if let _ = try? decoder.decode(CompletionStructure.self, from: data) {
        return { data in
            let completionStruct = try JSONDecoder().decode(CompletionStructure.self, from: data)
            return "\(completionStruct.prompt)\n\n\(completionStruct.completion)"
        }
    }
    
    throw LoRADataError.invalidJSON(line, 1)
}

func loadLines(url: URL) throws -> [String] {
    try String(contentsOf: url)
        .components(separatedBy: .newlines)
        .filter { !$0.isEmpty }
}

struct ChatStructure: Codable {
    struct Message: Codable {
        let role: String
        let content: String
    }
    let messages: [Message]
}

struct ToolStructure: Codable {
    let type: String
    let function: Function

    struct Function: Codable {
        let name: String
        let description: String
        let parameters: Parameters
    }

    struct Parameters: Codable {
        let type: String
        let properties: [String: Property]
        let required: [String]
    }

    struct Property: Codable {
        let type: String
        let description: String?
        let `enum`: [String]?

        enum CodingKeys: String, CodingKey {
            case type
            case description
            case `enum` = "enum"
        }
    }
}

struct CompletionStructure: Codable {
    let prompt: String
    let completion: String
}

struct TextStructure: Codable {
    let text: String
}
