// Copyright © 2026 Apple Inc.

import CoreVideo
import Foundation
import IOSurface

public struct Array2D<Element> {
    public let width: Int
    public let height: Int

    public private(set) var elements: [Element]

    init(width: Int, height: Int, initialValue: Element) {
        self.width = width
        self.height = height
        self.elements = Array(repeating: initialValue, count: width * height)
    }

    init(width: Int, height: Int, elements: [Element]) {
        self.width = width
        self.height = height
        self.elements = elements
    }

    private func index(_ x: Int, _ y: Int) -> Int {
        y * width + x
    }

    public subscript(x: Int, y: Int) -> Element {
        get { elements[index(x, y)] }
        set { elements[index(x, y)] = newValue }
    }

    @discardableResult
    public mutating func update(_ x: Int, _ y: Int, _ transform: (Element) -> Element) -> Element {
        let index = index(x, y)
        let new = transform(elements[index])
        elements[index] = new
        return new
    }

    public func map<Result>(_ transform: (Element) -> Result) -> Array2D<Result> {
        Array2D<Result>(
            width: width,
            height: height,
            elements: elements.map(transform)
        )
    }

    public func map<Result>(_ transform: (Element, Int, Int) -> Result) -> Array2D<Result> {
        Array2D<Result>(
            width: width,
            height: height,
            elements: elements.enumerated().map { (index, value) in
                let x = index % width
                let y = index / width
                return transform(value, x, y)
            }
        )
    }

}

extension Array2D where Element: BinaryInteger {
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.elements = Array(repeating: Element.zero, count: width * height)
    }
}

extension Array2D where Element: BinaryFloatingPoint {
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.elements = Array(repeating: Element.zero, count: width * height)
    }
}

public func applyLUT<Element: BinaryInteger>(
    _ input: Array2D<Element>, lut: [UInt32], max: Float, maxValue: UInt32
) -> Array2D<UInt32> {
    let lutCount = lut.count
    let scale = (Float(lutCount) - 1) / max
    let intMax = Int(max)

    let lut = lut.map {
        [($0 >> 24) & 0xff, ($0 >> 16) & 0xff, ($0 >> 8) & 0xff, ($0) & 0xff]
            .map(Int.init)
    }

    return input.map { value in
        if value == intMax {
            return maxValue
        }

        let index = Float(value) * scale
        let low = Int(floor(index))
        let high = min(low + 1, lutCount - 1)
        let fraction = index - Float(low)

        let colorLow = lut[low]
        let colorHigh = lut[high]

        func lerp(_ c: Int) -> Int {
            colorLow[c] + Int(fraction * Float(colorHigh[c] - colorLow[c]))
        }

        return UInt32(lerp(0) << 24 | lerp(1) << 16 | lerp(2) << 8 | lerp(3))
    }
}

public func createIOSurface(bgra: Array2D<UInt32>) -> IOSurface {
    let w = bgra.width
    let h = bgra.height

    // Create IOSurface and memcpy
    let bytesPerRow = w * 4
    let surface = IOSurface(properties: [
        .width: w,
        .height: h,
        .bytesPerElement: 4,
        .bytesPerRow: bytesPerRow,
        .pixelFormat: kCVPixelFormatType_32BGRA,
    ])!

    surface.lock(options: [], seed: nil as UnsafeMutablePointer<UInt32>?)
    _ = bgra.elements.withUnsafeBufferPointer { ptr in
        memcpy(surface.baseAddress, ptr.baseAddress, h * bytesPerRow)
    }
    surface.unlock(options: [], seed: nil as UnsafeMutablePointer<UInt32>?)

    return surface
}
