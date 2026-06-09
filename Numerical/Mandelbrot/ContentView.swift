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
            renderer.render()
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

            Spacer()

            fpsLabel(renderer: renderer)
        }
        .padding()
    }

    @ViewBuilder
    private func fpsLabel(renderer: Renderer) -> some View {
        if let fps = renderer.averageFPS, let frame = renderer.averageFrameTime {
            Text(String(format: "%.1f fps  (%.1f ms)", fps, frame * 1000))
                .monospacedDigit()
                .foregroundStyle(.secondary)
        } else {
            Text("— fps")
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
