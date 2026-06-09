// Copyright © 2026 Apple Inc.

import Charts
import MLX
import SwiftUI

struct DataPoint: Identifiable {
    let id: Int
    let x: Float
    let y: Float
    let series: String
}

struct ContentView: View {

    @State private var gradient = Gradient()
    @State private var step = 0
    @State private var points: [DataPoint] = []
    @State private var running = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Gradient Descent — Step \(step)")
                .font(.title2)

            Chart(points) { point in
                LineMark(
                    x: .value("x", point.x),
                    y: .value("y", point.y)
                )
                .foregroundStyle(by: .value("Series", point.series))
            }
            .chartForegroundStyleScale([
                "Actual": .blue,
                "Predicted": .orange,
            ])
            .chartXScale(domain: -2 ... 2)
            .chartYScale(domain: -5 ... 15)
            .frame(minHeight: 300)

            HStack {
                Button("Start") { startTraining() }
                    .disabled(running)
                Button("Reset") { reset() }
                    .disabled(running)
            }

            Text("θ = \(gradient.θ.description)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .onAppear { updatePoints() }
    }

    private func updatePoints() {
        let predY = model(gradient.θ, gradient.x)

        let xVals = gradient.x.asArray(Float.self)
        let aY = gradient.y.asArray(Float.self)
        let pY = predY.asArray(Float.self)

        var pts: [DataPoint] = []
        for i in xVals.indices {
            pts.append(DataPoint(id: i, x: xVals[i], y: aY[i], series: "Actual"))
            pts.append(DataPoint(id: i + xVals.count, x: xVals[i], y: pY[i], series: "Predicted"))
        }
        points = pts
    }

    private func startTraining() {
        running = true
        Task {
            for _ in 0 ..< gradient.totalSteps {
                gradient.step()
                step += 1
                updatePoints()
                try? await Task.sleep(for: .milliseconds(200))
            }
            running = false
        }
    }

    private func reset() {
        gradient = Gradient()
        step = 0
        updatePoints()
    }
}

#Preview {
    ContentView()
}
