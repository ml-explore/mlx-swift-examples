import Foundation
import LLM
import MLX

class DeviceStat: ObservableObject {
    @Published var gpuUsage: GPU.Snapshot
    private var initialGPUSnapshot: GPU.Snapshot
    private var timer: Timer?

    init() {
        initialGPUSnapshot = GPU.snapshot()
        gpuUsage = initialGPUSnapshot
        startTimer()
    }

    deinit {
        stopTimer()
    }

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateStats()
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func updateStats() {
        updateGPUUsages()
    }

    private func updateGPUUsages() {
        let gpuSnapshotDelta = initialGPUSnapshot.delta(GPU.snapshot())
        DispatchQueue.main.async { [weak self] in
            self?.gpuUsage = gpuSnapshotDelta
        }
    }

}
