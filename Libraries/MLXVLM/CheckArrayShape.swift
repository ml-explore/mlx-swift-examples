import MLX

/// Check if array is in a supported format for conv weights
public func checkArrayShape(_ arr: MLXArray) -> Bool {
    let shape = arr.shape
    switch shape.count {
    case 4:
        let outChannels = shape[0]
        let kH = shape[1]
        let kW = shape[2]
        // shape[3] is in_channels, which is ignored
        // Check if out_channels is the largest, and kH and kW are the same
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    case 3:
        let kW = shape[1]
        let outChannels = shape[2]
        // shape[0] is ignored
        // Check if kW is larger than or equal to out_channels
        return kW >= outChannels
    default:
        // Any other number of dimensions is not supported
        return false
    }
}
