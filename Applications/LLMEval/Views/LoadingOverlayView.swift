// Copyright © 2025 Apple Inc.

import SwiftUI

struct LoadingOverlayView: View {
    let modelInfo: String
    let downloadProgress: Double?
    let progressDescription: String?

    init(modelInfo: String, downloadProgress: Double? = nil, progressDescription: String? = nil) {
        self.modelInfo = modelInfo
        self.downloadProgress = downloadProgress
        self.progressDescription = progressDescription
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.4)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                if let progress = downloadProgress, progress < 1.0 {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                        .frame(width: 200)
                } else {
                    ProgressView()
                        .scaleEffect(1.5)
                        .progressViewStyle(.circular)
                }

                Text(modelInfo)
                    .font(.headline)
                    .foregroundStyle(.primary)

                if let description = progressDescription {
                    Text(description)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                Text(
                    "Models are large and may take a couple of minutes to download on first use. They are cached locally for faster loading in the future."
                )
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
            }
            .padding(32)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}
