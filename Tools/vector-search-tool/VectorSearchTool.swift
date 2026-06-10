import ArgumentParser
import Foundation
import TurboVec

/// Command-line demo of MLX-accelerated TurboQuant vector search.
///
/// Generates synthetic unit vectors, indexes them with TurboVec, runs top-k
/// search, and reports compression, latency, and recall vs brute-force cosine.
@main
struct VectorSearchTool: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Demonstrate MLX-accelerated TurboQuant vector search",
        discussion: """
            Builds a TurboQuant index over random vectors and compares approximate \
            search against an exact brute-force baseline. Enable MLX GPU acceleration \
            before indexing for faster batch rotation.
            """
    )

    @Option(name: .long, help: "Vector dimension (multiple of 8)")
    var dim: Int = 768

    @Option(name: .long, help: "Number of vectors to index")
    var count: Int = 10_000

    @Option(name: .long, help: "Number of search queries")
    var queries: Int = 100

    @Option(name: .long, help: "Top-k neighbors to retrieve")
    var k: Int = 10

    @Option(name: .long, help: "Quantization bit width (2, 3, or 4)")
    var bits: Int = 4

    @Flag(name: .long, help: "Enable MLX GPU acceleration via TurboVec")
    var gpu = false

    @Option(name: .long, help: "Write structured JSON benchmark report to path")
    var json: String?

    mutating func run() throws {
        guard dim > 0, dim % 8 == 0 else {
            throw ValidationError("--dim must be a positive multiple of 8")
        }
        guard let bitWidth = BitWidth(rawValue: UInt8(bits)) else {
            throw ValidationError("--bits must be 2, 3, or 4")
        }
        guard count > 0, queries > 0, k > 0 else {
            throw ValidationError("--count, --queries, and --k must be positive")
        }

        if gpu {
            MLXBackend.enableGPU()
            print("▸ MLX GPU acceleration enabled")
        } else {
            print("▸ CPU-only mode (pass --gpu to enable MLX acceleration)")
        }

        print("▸ Generating \(count) random unit vectors (d=\(dim))...")
        var rng = SplitMix64(seed: 42)
        let vectors = Self.generateVectors(n: count, dim: dim, rng: &rng)
        let queryVectors = Self.generateVectors(n: queries, dim: dim, rng: &rng)

        print("▸ Indexing with \(bits)-bit TurboQuant...")
        let index = TurboQuantIndex(dim: dim, bitWidth: bitWidth)
        let indexStart = CFAbsoluteTimeGetCurrent()
        try index.add(vectors)
        let indexMs = (CFAbsoluteTimeGetCurrent() - indexStart) * 1000

        let rawBytes = count * dim * MemoryLayout<Float>.size
        let indexBytes = index.indexSizeBytes
        let compression = Double(rawBytes) / Double(indexBytes)

        print("  ✓ Indexed in \(String(format: "%.1f", indexMs))ms")
        print("  ✓ Compression: \(String(format: "%.1f", compression))× (\(rawBytes) → \(indexBytes) bytes)")

        print("▸ Running \(queries) searches (k=\(k))...")
        var latencies = [Double]()
        var approximateResults = [[SearchHit]]()

        for query in queryVectors {
            let start = CFAbsoluteTimeGetCurrent()
            let hits = try index.search(query: query, k: k)
            latencies.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            approximateResults.append(hits)
        }

        let searchStats = Self.latencyStats(latencies)
        print("  ✓ Mean latency: \(String(format: "%.3f", searchStats.meanMs))ms")
        print("  ✓ P99 latency:  \(String(format: "%.3f", searchStats.p99Ms))ms")
        print("  ✓ QPS:          \(String(format: "%.0f", 1000.0 / searchStats.meanMs))")

        print("▸ Measuring recall vs brute-force baseline...")
        var recall1Sum = 0.0
        var recallKSum = 0.0
        var bfLatencies = [Double]()

        for (qi, query) in queryVectors.enumerated() {
            let start = CFAbsoluteTimeGetCurrent()
            let exact = BruteForceSearch.search(query: query, vectors: vectors, k: k)
            bfLatencies.append((CFAbsoluteTimeGetCurrent() - start) * 1000)

            let approx = approximateResults[qi]
            let exactTopK = Set(exact.prefix(k).map(\.index))
            let approxIndices = approx.prefix(k).map(\.index)

            if !exact.isEmpty, !approxIndices.isEmpty, approxIndices[0] == exact[0].index {
                recall1Sum += 1
            }
            let overlap = approxIndices.filter { exactTopK.contains($0) }.count
            recallKSum += Double(overlap) / Double(min(k, exact.count))
        }

        let recall1 = recall1Sum / Double(queries)
        let recallK = recallKSum / Double(queries)
        let bfStats = Self.latencyStats(bfLatencies)
        let speedup = bfStats.meanMs / searchStats.meanMs

        print("  ✓ Recall@1:  \(String(format: "%.4f", recall1))")
        print("  ✓ Recall@\(k): \(String(format: "%.4f", recallK))")
        print("  ✓ Brute-force mean: \(String(format: "%.3f", bfStats.meanMs))ms")
        print("  ✓ Speedup: \(String(format: "%.1f", speedup))×")

        let report = BenchmarkReport(
            hardware: Self.detectHardware(),
            config: BenchmarkReport.TestConfig(
                dimensions: dim,
                bitWidth: bits,
                numVectors: count,
                numQueries: queries,
                k: k
            ),
            search: BenchmarkReport.SearchResult(
                latency: searchStats,
                queriesPerSecond: 1000.0 / searchStats.meanMs,
                recallAt1: recall1,
                recallAtK: recallK
            ),
            memory: BenchmarkReport.MemoryUsage(
                rawBytes: rawBytes,
                indexBytes: indexBytes,
                compressionRatio: compression,
                bytesPerVector: Double(indexBytes) / Double(count)
            ),
            indexing: BenchmarkReport.IndexingResult(
                totalMs: indexMs,
                perVectorUs: indexMs * 1000.0 / Double(count),
                vectorsPerSecond: Double(count) / (indexMs / 1000.0)
            ),
            bruteForceBaseline: BenchmarkReport.SearchResult(
                latency: bfStats,
                queriesPerSecond: 1000.0 / bfStats.meanMs,
                recallAt1: 1.0,
                recallAtK: 1.0
            ),
            timestamp: ISO8601DateFormatter().string(from: Date()),
            version: "0.1.0"
        )

        print()
        print("Summary: \(String(format: "%.1f", compression))× compression, " +
            "R@1=\(String(format: "%.4f", recall1)), " +
            "\(String(format: "%.1f", speedup))× faster than brute-force")

        let jsonText = try report.jsonString()
        if let json {
            try jsonText.write(toFile: json, atomically: true, encoding: .utf8)
            print("Report saved to \(json)")
        }

        print()
        print("--- JSON Report ---")
        print(jsonText)
    }

    private static func generateVectors(n: Int, dim: Int, rng: inout SplitMix64) -> [[Float]] {
        (0..<n).map { _ in
            var vector = (0..<dim).map { _ -> Float in
                let u1 = max(Float(rng.nextUniform()), Float.leastNormalMagnitude)
                let u2 = Float(rng.nextUniform())
                return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
            }
            VectorMath.normalize(&vector)
            return vector
        }
    }

    private static func latencyStats(_ values: [Double]) -> BenchmarkReport.LatencyStats {
        let sorted = values.sorted()
        let n = sorted.count
        return BenchmarkReport.LatencyStats(
            p50Ms: sorted[n / 2],
            p95Ms: sorted[Int(Double(n) * 0.95)],
            p99Ms: sorted[Int(Double(n) * 0.99)],
            meanMs: sorted.reduce(0, +) / Double(n),
            minMs: sorted.first ?? 0,
            maxMs: sorted.last ?? 0
        )
    }

    private static func detectHardware() -> BenchmarkReport.HardwareInfo {
        var chip = "Unknown"
        #if arch(arm64)
        chip = sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"
        #else
        chip = sysctlString("machdep.cpu.brand_string") ?? "x86_64"
        #endif

        return BenchmarkReport.HardwareInfo(
            chip: chip,
            cores: ProcessInfo.processInfo.activeProcessorCount,
            memoryGB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)),
            os: ProcessInfo.processInfo.operatingSystemVersionString
        )
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        sysctlbyname(name, nil, &size, nil, 0)
        guard size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname(name, &buffer, &size, nil, 0)
        return String(cString: buffer)
    }
}
