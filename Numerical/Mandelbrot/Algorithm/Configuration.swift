// Copyright © 2026 Apple Inc.

import Foundation

public struct Configuration: Sendable {
    public var width: Int = 1280
    public var height: Int = 1024
    public var maxIterations: Int = 500
    public var centerX: Float = -0.5
    public var centerY: Float = 0.0
    public var zoom: Float = 1.0
    public var escapeRadius: Float = 2.0

    public var escapeRadiusSquared: Float { escapeRadius * escapeRadius }
    public var aspect: Float { Float(width) / Float(height) }
    public var rangeX: Float { 3.5 / zoom }
    public var rangeY: Float { rangeX / aspect }
    public var xMin: Float { centerX - rangeX / 2 }
    public var xMax: Float { centerX + rangeX / 2 }
    public var yMin: Float { centerY - rangeY / 2 }
    public var yMax: Float { centerY + rangeY / 2 }
    public var xStep: Float { rangeX / Float(width) }
    public var yStep: Float { rangeY / Float(height) }

    public func lerp(other: Configuration, steps: Int, step: Int) -> Configuration {
        func lerp(_ a: Float, _ b: Float) -> Float {
            a + (b - a) / Float(steps) * Float(step)
        }

        return .init(
            centerX: lerp(centerX, other.centerX),
            centerY: lerp(centerY, other.centerY),
            zoom: lerp(zoom, other.zoom),
        )
    }
}

let lut: [UInt32] = [
    0xFF00_0000,
    0xFFFF_FFFF,
    0xFFFF_0000,
    0xFFFF_FF00,
    0xFF00_FF00,
    0xFF00_FFFF,
    0xFF00_00FF,
    0xFFFF_FFFF,
    0xFF00_00FF,
    0xFF00_FF00,
    0xFF00_FFFF,
    0xFFFF_0000,
    0xFFFF_FF00,
    0xFFFF_FFFF,
]
