import ArgumentParser
import MLX

/// Argument package for adjusting and reporting GPU memory usage.
struct MemoryArguments: ParsableArguments, Sendable {

    @Flag(name: .long, help: "Show GPU memory stats before exit.")
    var memoryStats = false

    @Option(name: .long, help: "Maximum GPU cache size in megabytes.")
    var cacheSize: Int?

    @Option(name: .long, help: "Maximum GPU memory size in megabytes.")
    var memorySize: Int?

    private(set) var startMemory: GPU.Snapshot?

    mutating func start<L>(_ operation: @Sendable () async throws -> L) async throws -> L {
        applyLimits()
        let result = try await operation()
        startMemory = GPU.snapshot()
        return result
    }

    mutating func start() {
        applyLimits()
        startMemory = GPU.snapshot()
    }

    func reportCurrent() {
        guard memoryStats else { return }
        let memory = GPU.snapshot()
        print(memory.description)
    }

    func reportMemoryStatistics() {
        guard memoryStats, let startMemory else { return }

        let endMemory = GPU.snapshot()

        print("=======")
        print("GPU memory limit: \(GPU.memoryLimit / 1024)K")
        print("GPU cache limit:  \(GPU.cacheLimit / 1024)K")
        print("")
        print("=======")
        print("Starting snapshot")
        print(startMemory.description)
        print("")
        print("=======")
        print("Ending snapshot")
        print(endMemory.description)
        print("")
        print("=======")
        print("Delta")
        print(startMemory.delta(endMemory).description)
    }

    private func applyLimits() {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }
    }
}
