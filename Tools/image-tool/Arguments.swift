// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import MLX

#if swift(>=5.10)
    /// Extension to allow URL command line arguments.
    extension URL: @retroactive ExpressibleByArgument {
        public init?(argument: String) {
            if argument.contains("://") {
                self.init(string: argument)
            } else {
                self.init(filePath: argument)
            }
        }
    }
#else
    /// Extension to allow URL command line arguments.
    extension URL: ExpressibleByArgument {
        public init?(argument: String) {
            if argument.contains("://") {
                self.init(string: argument)
            } else {
                self.init(filePath: argument)
            }
        }
    }
#endif

/// Argument package for adjusting and reporting memory use.
struct MemoryArguments: ParsableArguments, Sendable {

    @Flag(name: .long, help: "Show memory stats")
    var memoryStats = false

    @Option(name: .long, help: "Maximum cache size in M")
    var cacheSize = 1024

    @Option(name: .long, help: "Maximum memory size in M")
    var memorySize: Int?

    var startMemory: Memory.Snapshot?

    mutating func start<L>(_ load: () async throws -> L) async throws -> L {
        Memory.cacheLimit = cacheSize * 1024 * 1024

        if let memorySize {
            Memory.memoryLimit = memorySize * 1024 * 1024
        }

        let result = try await load()
        startMemory = Memory.snapshot()

        return result
    }

    mutating func start() {
        Memory.cacheLimit = cacheSize * 1024 * 1024

        if let memorySize {
            Memory.memoryLimit = memorySize * 1024 * 1024
        }

        startMemory = Memory.snapshot()
    }

    func reportCurrent() {
        if memoryStats {
            let memory = Memory.snapshot()
            print(memory.description)
        }
    }

    func reportMemoryStatistics() {
        if memoryStats, let startMemory {
            let endMemory = Memory.snapshot()

            print("=======")
            print("Memory size: \(Memory.memoryLimit / 1024)K")
            print("Cache size:  \(Memory.cacheLimit / 1024)K")

            print("")
            print("=======")
            print("Starting memory")
            print(startMemory.description)

            print("")
            print("=======")
            print("Ending memory")
            print(endMemory.description)

            print("")
            print("=======")
            print("Growth")
            print(startMemory.delta(endMemory).description)

        }
    }
}
