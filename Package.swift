// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-libraries",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "LLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MNIST",
            targets: ["MLXMNIST"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.5"),
        .package(url: "https://github.com/1024jp/GzipSwift", from: "6.0.1"),
        .package(url: "https://github.com/apple/swift-async-algorithms", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
            ],
            path: "Libraries/LLM",
            exclude: [
                "README.md",
                "LLM.h",
            ]
        ),
        .target(
            name: "MLXMNIST",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
                .product(name: "Gzip", package: "GzipSwift"),
            ],
            path: "Libraries/MNIST",
            exclude: [
                "README.md",
                "MNIST.h",
            ]
        ),
    ]
)
