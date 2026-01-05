// Copyright © 2025 Apple Inc.

import MLX
import SwiftUI

struct MetricsView: View {
    let tokensPerSecond: Double
    let timeToFirstToken: Double
    let promptLength: Int
    let totalTokens: Int
    let totalTime: Double
    let memoryUsed: Int
    let cacheMemory: Int
    let peakMemory: Int

    @State private var showMemoryDetails = false

    @Environment(\.horizontalSizeClass) var horizontalSizeClass

    var body: some View {
        if horizontalSizeClass == .compact {
            DisclosureGroup("Statistics") {
                stats
                    .scaleEffect(0.8)
            }
        } else {
            stats
        }
    }

    var stats: some View {
        VStack(spacing: 12) {
            // Top row
            HStack(spacing: 12) {
                MetricCard(
                    icon: "speedometer",
                    title: "Tokens/sec",
                    value: String(format: "%.1f", tokensPerSecond)
                )
                MetricCard(
                    icon: "timer",
                    title: "Time to First Token",
                    value: String(format: "%.0fms", timeToFirstToken)
                )
                MetricCard(
                    icon: "text.alignleft",
                    title: "Prompt Length",
                    value: "\(promptLength)"
                )
            }

            // Bottom row
            HStack(spacing: 12) {
                MetricCard(
                    icon: "number",
                    title: "Total Tokens",
                    value: "\(totalTokens)"
                )
                MetricCard(
                    icon: "hourglass",
                    title: "Total Time",
                    value: String(format: "%.1fs", totalTime)
                )
                ZStack(alignment: .topTrailing) {
                    MetricCard(
                        icon: "memorychip",
                        title: "Memory",
                        value: FormatUtilities.formatMemory(memoryUsed)
                    )
                    Button(action: {
                        #if os(iOS)
                            showMemoryDetails = true
                        #endif
                    }) {
                        Image(systemName: "info.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(width: 44, height: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(
                        """
                        Active Memory: \(FormatUtilities.formatMemory(memoryUsed))/\(FormatUtilities.formatMemory(GPU.memoryLimit))
                        Cache Memory: \(FormatUtilities.formatMemory(cacheMemory))/\(FormatUtilities.formatMemory(GPU.cacheLimit))
                        Peak Memory: \(FormatUtilities.formatMemory(peakMemory))
                        """
                    )
                }
            }
        }
        .padding(.top, 8)
        .alert("Memory Details", isPresented: $showMemoryDetails) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(
                """
                Active Memory: \(FormatUtilities.formatMemory(memoryUsed))/\(FormatUtilities.formatMemory(GPU.memoryLimit))
                Cache Memory: \(FormatUtilities.formatMemory(cacheMemory))/\(FormatUtilities.formatMemory(GPU.cacheLimit))
                Peak Memory: \(FormatUtilities.formatMemory(peakMemory))
                """)
        }
    }
}
