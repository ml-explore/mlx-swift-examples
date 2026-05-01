//
//  DownloadProgressView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import SwiftUI

struct DownloadProgressView: View {
    let progress: Progress

    @State private var isShowingDownload = false

    private var bytesText: String {
        let completed = progress.completedUnitCount.formatted(.byteCount(style: .file))
        let total = progress.totalUnitCount.formatted(.byteCount(style: .file))
        return "\(completed) of \(total)"
    }

    private var percentText: String {
        progress.fractionCompleted.formatted(.percent.precision(.fractionLength(0)))
    }

    var body: some View {
        Button {
            isShowingDownload = true
        } label: {
            Image(systemName: "arrow.down")
                .foregroundStyle(.tint)
        }
        .popover(isPresented: $isShowingDownload, arrowEdge: .bottom) {
            VStack {
                ProgressView(value: progress.fractionCompleted) {
                    HStack {
                        Text(bytesText)
                            .bold()
                        Spacer()
                        Text(percentText)
                    }
                }

                Text("Downloading...")
                    .padding(.horizontal, 32)
            }
            .padding()
        }
    }
}

#Preview {
    DownloadProgressView(progress: Progress(totalUnitCount: 6))
}
