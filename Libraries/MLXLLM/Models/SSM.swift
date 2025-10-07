//
//  SSM.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/10/01.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public func computeDt(_ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float))
    -> MLXArray
{
    let dt = softplus(dt + dtBias)
    return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)
}

private func makeSSMKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = Ds / 32;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            // C and B have shape [batch, group, state_dim]
            // C and B need to be offset by group size
            auto C_ = C + g_idx * Ds;
            auto B_ = B + g_idx * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            auto dt_ = static_cast<float>(dt[n]);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto idx = d_idx * Ds + s_idx;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                auto state = dA * i_state[idx] + dB_by_x;
                o_state[idx] = static_cast<T>(state);
                acc += state * C_[s_idx];
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
            }
        """

    return MLXFast.metalKernel(
        name: "ssm_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: source
    )
}

private final class SSMKernelManager: @unchecked Sendable {
    static let shared = SSMKernelManager()

    let ssmKernel: MLXFast.MLXFastKernel?

    private init() {
        ssmKernel = makeSSMKernel()
    }
}

func ssmUpdateKernel(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    let (n, _, h, d) = hiddenStates.shape4
    let inputType = hiddenStates.dtype
    let (hb, ds) = (B.dim(-2), B.dim(-1))

    let dt = computeDt(dt, dtBias, timeStepLimit)

    guard let kernel = SSMKernelManager.shared.ssmKernel else {
        fatalError("SSM kernel not available")
    }

    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, dt, state],
        template: [
            ("T", inputType),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h / hb),
        ],
        grid: (32, d, h * n),
        threadGroup: (32, 8, 1),
        outputShapes: [[n, 1, h, d], state.shape],
        outputDTypes: [inputType, inputType]
    )

    return (outputs[0], outputs[1])
}

public func segsum(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let l = x.dim(-1)
    var x = x

    if let mask = mask {
        let mask = MLX.expandedDimensions(mask, axis: 1)
        x = x * mask
    }

    x = MLX.repeated(x[.ellipsis, .newAxis], count: l, axis: -1)
    x = MLX.tril(x, k: -1)
    var xSegsum = MLX.cumsum(x, axis: -2)

    if let mask = mask {
        xSegsum = which(
            mask[.ellipsis, .newAxis, 0...] * mask[.ellipsis, .newAxis],
            xSegsum,
            MLXArray(-Float.infinity)
        )
    }

    return xSegsum
}

public func ssmAttn(
    x: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let (_, _, g, d) = B.shape4

    let dt = computeDt(dt, dtBias, timeStepLimit)
    let repeats = h / g
    let A = -MLX.exp(ALog)
    var B = MLX.transposed(B, axes: [0, 2, 3, 1])

    // A * s + B * C
    var CB = MLX.swappedAxes(C, 1, 2).matmul(B)
    CB = MLX.repeated(CB, count: repeats, axis: 1)

    let dtA = dt * A.reshaped(1, 1, -1)
    var decay = MLX.exp(segsum(dtA.swappedAxes(1, 2), mask: mask))

    let surrogateAttentionMatrix = MLX.tril(CB * decay, k: 0)

    let dtx = dt.reshaped(b, l, h, 1) * x
    var y = surrogateAttentionMatrix.matmul(dtx.swappedAxes(1, 2))
    y = MLX.swappedAxes(y, 1, 2)

    decay = decay[0..., 0..., (-1)..., 0...].transposed(0, 3, 1, 2)
    B = MLX.repeated(B, count: h / g, axis: 1).swappedAxes(2, 3)
    var dtxdecay = dtx * decay
    dtxdecay = dtxdecay.swappedAxes(1, 2).swappedAxes(2, 3)

    var nextState = dtxdecay.matmul(B)

    if var state = state {
        let expDtACumsum = MLX.exp(MLX.cumsum(dtA, axis: -2))
        nextState = nextState + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * state
        state = state.reshaped(b, 1, g, repeats, dh, d)
        let C = C.reshaped(b, l, g, 1, d, 1)
        let yPrev = (state.matmul(C)).squeezed(axis: -1).flattened(start: 2, end: 3)
        y = y + expDtACumsum[.ellipsis, .newAxis] * yPrev
    }

    y = y + x * D.reshaped(1, 1, h, 1)
    return (y, nextState)
}

public func ssmUpdate(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let seqLen = hiddenStates.dim(1)

    if seqLen == 1,
        let state = state,
        SSMKernelManager.shared.ssmKernel != nil
    {
        return ssmUpdateKernel(
            hiddenStates: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit
        )
    } else {
        return ssmAttn(
            x: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit,
            mask: mask
        )
    }
}
