// Copyright © 2024 Apple Inc.

import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

enum ImageError: LocalizedError {
    case failedToSave
    case unableToOpen

    var errorDescription: String? {
        switch self {
        case .failedToSave:
            return String(localized: "Failed to save the image to the specified location.")
        case .unableToOpen:
            return String(localized: "Unable to open the image file.")
        }
    }
}

/// Conversion utilities for moving between `MLXArray`, `CGImage` and files.
public struct Image {

    public let data: MLXArray

    /// Create an Image from a MLXArray with ndim == 3
    public init(_ data: MLXArray) {
        precondition(data.ndim == 3)
        self.data = data
    }

    /// Create an Image by loading from a file
    public init(url: URL, maximumEdge: Int? = nil) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImageError.unableToOpen
        }

        self.init(image: image)
    }

    /// Create an image from a CGImage
    public init(image: CGImage, maximumEdge: Int? = nil) {
        // ensure the sizes ar multiples of 64 -- this doesn't worry about
        // the aspect ratio

        var width = image.width
        var height = image.height

        if let maximumEdge {
            func scale(_ edge: Int, _ maxEdge: Int) -> Int {
                Int(round(Float(maximumEdge) / Float(maxEdge) * Float(edge)))
            }

            // aspect fit inside the given maximum
            if width >= height {
                width = scale(width, image.width)
                height = scale(height, image.width)
            } else {
                width = scale(width, image.height)
                height = scale(height, image.height)
            }
        }

        // size must be multiples of 64 -- coerce without regard to aspect ratio
        width = width - width % 64
        height = height - height % 64

        var raster = Data(count: width * 4 * height)
        raster.withUnsafeMutableBytes { ptr in
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            let context = CGContext(
                data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue)!

            context.draw(
                image, in: CGRect(origin: .zero, size: .init(width: width, height: height)))
        }

        self.data = MLXArray(raster, [height, width, 4], type: UInt8.self)[0..., 0..., ..<3]
    }

    /// Convert the image data to a CGImage
    public func asCGImage() -> CGImage {
        var raster = data

        // we need 4 bytes per pixel
        if data.dim(-1) == 3 {
            raster = padded(raster, widths: [0, 0, [0, 1]])
        }

        class DataHolder {
            var data: Data
            init(_ data: Data) {
                self.data = data
            }
        }

        let holder = DataHolder(raster.asData(access: .copy).data)

        let payload = Unmanaged.passRetained(holder).toOpaque()
        func release(payload: UnsafeMutableRawPointer?, data: UnsafeMutableRawPointer?) {
            Unmanaged<DataHolder>.fromOpaque(payload!).release()
        }

        return holder.data.withUnsafeMutableBytes { ptr in
            let (H, W, C) = raster.shape3
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!

            let context = CGContext(
                data: ptr.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * C,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue, releaseCallback: release,
                releaseInfo: payload)!
            return context.makeImage()!
        }
    }

    /// Convert the image data to a CIImage
    public func asCIImage() -> CIImage {
        // we need 4 bytes per pixel
        var raster = data
        if data.dim(-1) == 3 {
            raster = padded(raster, widths: [0, 0, [0, 1]], value: MLXArray(255))
        }

        let arrayData = raster.asData()
        let (H, W, C) = raster.shape3
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!

        return CIImage(
            bitmapData: arrayData.data, bytesPerRow: W * 4, size: .init(width: W, height: H),
            format: .RGBA8, colorSpace: cs)
    }

    /// Save the image
    public func save(url: URL) throws {
        let uti = UTType(filenameExtension: url.pathExtension) ?? UTType.png

        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, uti.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, asCGImage(), nil)
        if !CGImageDestinationFinalize(destination) {
            throw ImageError.failedToSave
        }
    }
}
