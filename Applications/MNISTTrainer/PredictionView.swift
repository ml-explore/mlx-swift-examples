//
//  PredictionView.swift
//  MNISTTrainer
//
//  Created by Rounak Jain on 3/9/24.
//

import MLX
import MLXMNIST
import MLXNN
import SwiftUI

struct Canvas: View {

    @Binding var path: Path
    @State var lastPoint: CGPoint?

    var body: some View {
        path
            .stroke(.white, lineWidth: 10)
            .background(.black)
            .gesture(
                DragGesture(minimumDistance: 0.05)
                    .onChanged { touch in
                        add(point: touch.location)
                    }
                    .onEnded { touch in
                        lastPoint = nil
                    }
            )
    }

    func add(point: CGPoint) {
        var newPath = path
        if let lastPoint {
            newPath.move(to: lastPoint)
            newPath.addLine(to: point)
        } else {
            newPath.move(to: point)
        }
        self.path = newPath
        lastPoint = point
    }
}

extension Path {
    mutating func center(to newMidPoint: CGPoint) {
        let middleX = boundingRect.midX
        let middleY = boundingRect.midY
        self = offsetBy(dx: newMidPoint.x - middleX, dy: newMidPoint.y - middleY)
    }
}

struct PredictionView: View {
    @State var path: Path = Path()
    @State var prediction: Int?
    let model: LeNetContainer
    let canvasSize = 150.0
    let mnistImageSize: CGSize = CGSize(width: 28, height: 28)

    var body: some View {
        VStack {
            if let prediction {
                Text("You've drawn a \(prediction)")
            } else {
                Text("Draw a digit")
            }
            Canvas(path: $path)
                .frame(width: canvasSize, height: canvasSize)
            HStack {
                Button("Predict") {
                    path.center(to: CGPoint(x: canvasSize / 2, y: canvasSize / 2))
                    predict()
                }
                Button("Clear") {
                    path = Path()
                    prediction = nil
                }
            }
        }
    }

    @MainActor
    func predict() {
        let imageRenderer = ImageRenderer(
            content: Canvas(path: $path).frame(width: 150, height: 150))

        if let image = imageRenderer.cgImage {
            Task {
                self.prediction = await model.evaluate(image: image)
            }
        }
    }
}

extension CGImage {
    func grayscaleImage(with newSize: CGSize) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: Int(newSize.width),
                height: Int(newSize.height),
                bitsPerComponent: 8,
                bytesPerRow: Int(newSize.width),
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue)
        else {
            return nil
        }
        context.draw(self, in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.width))
        return context.makeImage()
    }

    func pixelData() -> MLXArray {
        guard let data = self.dataProvider?.data else {
            return []
        }
        let bytePtr = CFDataGetBytePtr(data)
        let count = CFDataGetLength(data)
        return MLXArray(UnsafeBufferPointer(start: bytePtr, count: count))
    }
}
