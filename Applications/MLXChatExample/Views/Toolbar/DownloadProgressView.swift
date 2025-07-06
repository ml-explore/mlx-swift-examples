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

    var body: some View {
        Button {
            isShowingDownload = true
        } label: {
            Image(systemName: "arrow.down.square")
                .foregroundStyle(.tint)
        }
        .popover(isPresented: $isShowingDownload, arrowEdge: .bottom) {
            VStack {
                ProgressView(value: progress.fractionCompleted) {
                    HStack {
                        Text(progress.localizedAdditionalDescription)
                            .bold()
                        Spacer()
                        Text(progress.localizedDescription)
                    }
                }

                Text("The model is downloading")
                    .padding(.horizontal, 32)
            }
            .padding()
        }
    }
}

#Preview {
    DownloadProgressView(progress: Progress(totalUnitCount: 6))
}
