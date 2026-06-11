// Copyright © 2026 Apple Inc.

import Foundation
import MLX

public struct Configuration {
    public var width: Int = 1280
    public var height: Int = 1024

    public var initialTemperature: MLXArray

    public struct Wall: Sendable {
        var x: Int
        var y: Int
        var width: Int
        var height: Int

        var minX: Int { x }
        var maxX: Int { x + width }
        var minY: Int { y }
        var maxY: Int { y + height }
    }
    public var walls: [Wall]

    public struct HeatSource: Sendable {
        var x: Int
        var y: Int
        var radius: Int
        var temperature: Float
    }
    public var heatSources: [HeatSource]

    public init(
        width: Int = 1280, height: Int = 1024, initialTemperature: MLXArray? = nil,
        walls: [Wall] = [], heatSources: [HeatSource] = []
    ) {
        self.width = width
        self.height = height
        self.initialTemperature = initialTemperature ?? zeros([height, width], dtype: .float16)
        self.walls = walls
        self.heatSources = heatSources
    }

    public mutating func addRandomWall() {
        enum Kind: CaseIterable {
            case horizontal
            case vertical
            case horizontalSplit
            case verticalSplit
        }

        let s = min(width, height)
        let length = Int.random(in: (s / 6) ... (3 * s / 4))
        let thickness = Int.random(in: 4 ... 10)
        let border = thickness * 3

        switch Kind.allCases.randomElement()! {
        case .horizontal:
            walls.append(
                .init(
                    x: Int.random(in: border ... (width - border - length)),
                    y: Int.random(in: border ... (height - border)),
                    width: length,
                    height: thickness)
            )
        case .vertical:
            walls.append(
                .init(
                    x: Int.random(in: border ... (width - border)),
                    y: Int.random(in: border ... (height - border - length)),
                    width: thickness,
                    height: length)
            )

        case .horizontalSplit:
            let f = Float.random(in: 0.25 ... 0.75)
            let wall = Wall(
                x: Int.random(in: border ... (width - border - length)),
                y: Int.random(in: border ... (height - border)),
                width: Int(Float(length) * (f - 0.05)),
                height: thickness)
            let wall2 = Wall(
                x: wall.x + wall.width + length / 10,
                y: wall.y,
                width: Int(Float(length) * (1 - f - 0.05)),
                height: thickness)
            walls.append(wall)
            walls.append(wall2)

        case .verticalSplit:
            let f = Float.random(in: 0.25 ... 0.75)
            let wall = Wall(
                x: Int.random(in: border ... (width - border)),
                y: Int.random(in: border ... (height - border - length)),
                width: thickness,
                height: Int(Float(length) * (f - 0.05)))
            let wall2 = Wall(
                x: wall.x,
                y: wall.y + wall.height + length / 10,
                width: thickness,
                height: Int(Float(length) * (1 - f - 0.05)))
            walls.append(wall)
            walls.append(wall2)
        }
    }

    public mutating func addRandomHeatSource() {
        heatSources.append(
            .init(
                x: Int.random(in: width / 10 ... 9 * width / 10),
                y: Int.random(in: height / 10 ... 9 * height / 10),
                radius: Int.random(in: 20 ... 40), temperature: Float.random(in: 0.8 ... 1.1))
        )
    }

    public func asRoom() -> Room {
        let rows = arange(0, height)[.ellipsis, .newAxis]
        let cols = arange(0, width)[.newAxis, .ellipsis]

        var wallMask =
            ((rows .== 0) .|| (rows .== (height - 1)) .|| (cols .== 0) .|| (cols .== (width - 1)))

        for wall in walls {
            wallMask = wallMask
                .|| ((rows .>= wall.minY) .&& (rows .< wall.maxY) .&& (cols .>= wall.minX)
                    .&& (cols .< wall.maxX))
        }

        var heatSources = zeros([height, width])
        for heatSource in self.heatSources {
            let dx = cols - heatSource.x
            let dy = rows - heatSource.y
            let mask = (dx * dx + dy * dy) .< (heatSource.radius * heatSource.radius)
            heatSources = which(mask, heatSource.temperature, heatSources)
        }

        let staticMask = wallMask .|| (heatSources .> 0)

        let redMask = ((rows + cols) % 2) .== 0
        let blackMask = ((rows + cols) % 2) .== 1

        return Room(
            temperature: initialTemperature,
            heatSources: heatSources,
            staticMask: staticMask,
            wallMask: wallMask,
            redMask: redMask, blackMask: blackMask)
    }
}

public struct Room {
    // [H, W]
    var temperature: MLXArray
    var heatSources: MLXArray
    var staticMask: MLXArray

    // [H, W] -- for display
    var wallMask: MLXArray

    // [H, W] -- for SOR
    var redMask: MLXArray
    var blackMask: MLXArray
}

let lut: [UInt32] = [
    0xFF00_0000,
    0xFF80_0000,
    0xFFFF_0000,
    0xFFFF_FF00,
    0xFFFF_FFFF,
]
