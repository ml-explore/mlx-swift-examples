// Copyright © 2026 Apple Inc.

import SwiftUI

struct ContentView: View {
    @State private var renderer = Renderer()

    var body: some View {
        @Bindable var renderer = renderer

        VStack(spacing: 0) {
            ZStack {
                if let image = renderer.image {
                    ImageView(image: image)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Color.black
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            controls(renderer: renderer)
        }
        .ignoresSafeArea(edges: .top)
        .task {
            renderer.start()
        }
    }

    @ViewBuilder
    private func controls(renderer: Renderer) -> some View {
        @Bindable var renderer = renderer

        HStack(spacing: 16) {
            Picker("Renderer", selection: $renderer.kind) {
                ForEach(RendererKind.allCases) { kind in
                    Text(kind.rawValue).tag(kind)
                }
            }
            .pickerStyle(.segmented)
            .fixedSize()

            Button(renderer.isAnimating ? "Stop" : "Start") {
                if renderer.isAnimating {
                    renderer.stopAnimation()
                } else {
                    renderer.startAnimation()
                }
            }

            Button("Rebuild") {
                renderer.rebuildRoom()
            }

            Spacer()

            fpsLabel(renderer: renderer)
        }
        .padding()
    }

    @ViewBuilder
    private func fpsLabel(renderer: Renderer) -> some View {
        if let fps = renderer.averageFPS, let frame = renderer.averageFrameTime {
            if frame * 1000 < 10 {
                Text(String(format: "%.0f max fps  (%.3f ms)", fps, frame * 1000))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            } else {
                Text(String(format: "%.0f max fps  (%.0f ms)", fps, frame * 1000))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        } else {
            Text("— max fps")
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
