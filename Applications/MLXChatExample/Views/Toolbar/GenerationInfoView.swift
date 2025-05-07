//
//  GenerationInfoView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import SwiftUI

struct GenerationInfoView: View {
    let tokensPerSecond: Double
    let timeToFirstToken: Double

    var body: some View {
        HStack {
            if timeToFirstToken > 0 {
                Text(String(format: "TTFT: %.2f s", timeToFirstToken))
            }
            if tokensPerSecond > 0 {
                Text(String(format: "TPS: %.2f", tokensPerSecond))
            }
        }
        .lineLimit(1)
        .frame(minWidth: 150, alignment: .leading)
    }
}

#Preview {
    GenerationInfoView(tokensPerSecond: 58.5834, timeToFirstToken: 1.234)
        .padding()
}
