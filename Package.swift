// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-libraries",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "MLXMNIST",
            targets: ["MLXMNIST"]),
        .library(
            name: "StableDiffusion",
            targets: ["StableDiffusion"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/offlyn-ai/mlx-turbovec-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),

        // Note: used by StableDiffusion library to download weights
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMajor(from: "1.3.0")
        ),
        .package(url: "https://github.com/1024jp/GzipSwift", "6.0.1" ... "6.0.1"),  // Only needed by MLXMNIST
    ],
    targets: [
        .target(
            name: "MLXMNIST",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Gzip", package: "GzipSwift"),
            ],
            path: "Libraries/MLXMNIST",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "StableDiffusion",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Libraries/StableDiffusion",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "vector-search-tool",
            dependencies: [
                .product(name: "TurboVec", package: "mlx-turbovec-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/vector-search-tool",
            exclude: [
                "README.md"
            ]
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    // docc builder
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
