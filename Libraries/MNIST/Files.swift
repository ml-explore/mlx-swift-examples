// Copyright © 2024 Apple Inc.

import Foundation
import Gzip
import MLX

// based on https://github.com/ml-explore/mlx-examples/blob/main/mnist/mnist.py

public enum Use: String, Hashable, Sendable {
    case test
    case training
}

public enum DataKind: String, Hashable, Sendable {
    case images
    case labels
}

public struct FileKind: Hashable, CustomStringConvertible, Sendable {
    let use: Use
    let data: DataKind

    public init(_ use: Use, _ data: DataKind) {
        self.use = use
        self.data = data
    }

    public var description: String {
        "\(use.rawValue)-\(data.rawValue)"
    }
}

struct LoadInfo: Sendable {
    let name: String
    let offset: Int
    let convert: @Sendable (MLXArray) -> MLXArray
}

let baseURL = URL(string: "https://raw.githubusercontent.com/fgnt/mnist/master/")!

private let files = [
    FileKind(.training, .images): LoadInfo(
        name: "train-images-idx3-ubyte.gz",
        offset: 16,
        convert: {
            $0.reshaped([-1, 28, 28, 1]).asType(.float32) / 255.0
        }),
    FileKind(.test, .images): LoadInfo(
        name: "t10k-images-idx3-ubyte.gz",
        offset: 16,
        convert: {
            $0.reshaped([-1, 28, 28, 1]).asType(.float32) / 255.0
        }),
    FileKind(.training, .labels): LoadInfo(
        name: "train-labels-idx1-ubyte.gz",
        offset: 8,
        convert: {
            $0.asType(.uint32)
        }),
    FileKind(.test, .labels): LoadInfo(
        name: "t10k-labels-idx1-ubyte.gz",
        offset: 8,
        convert: {
            $0.asType(.uint32)
        }),
]

public func download(into: URL) async throws {
    for (_, info) in files {
        let fileURL = into.appending(component: info.name)
        if !FileManager.default.fileExists(atPath: fileURL.path()) {
            print("Download: \(info.name)")
            let url = baseURL.appending(component: info.name)
            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                fatalError("Unable to download \(url), not an http response: \(response)")
            }
            guard httpResponse.statusCode == 200 else {
                fatalError("Unable to download \(url): \(httpResponse)")
            }

            try data.write(to: fileURL)
        }
    }
}

public func load(from: URL) throws -> [FileKind: MLXArray] {
    var result = [FileKind: MLXArray]()

    for (key, info) in files {
        let fileURL = from.appending(component: info.name)
        let data = try Data(contentsOf: fileURL).gunzipped()

        let array = MLXArray(
            data.dropFirst(info.offset), [data.count - info.offset], type: UInt8.self)

        result[key] = info.convert(array)
    }

    return result
}
