Build me a command line chat tool that uses MLX and Swift. More details below... 

# MLX SWIFT FRAMEWORK - COMPREHENSIVE DOCUMENTATION AND API REFERENCE

This document provides complete documentation for MLX Swift, a Swift API for Apple's MLX machine learning framework. Use this documentation to understand how to build high-performance machine learning applications on Apple Silicon.

## TABLE OF CONTENTS

1. Introduction to MLX Swift
2. Core Concepts and Architecture
3. MLXArray - The Fundamental Data Structure
4. Operations and Transformations
5. Neural Network Module (MLX.NN)
6. Language Model Support
7. Memory Management and Performance
8. Complete API Reference
9. Code Examples

---

## 1. INTRODUCTION TO MLX SWIFT

MLX Swift is the Swift API for MLX, a machine learning framework designed specifically for Apple Silicon. It provides:

- **Lazy evaluation**: Operations are compiled and executed only when needed
- **Unified memory**: Leverages Apple Silicon's unified memory architecture
- **Multi-device support**: Seamlessly uses CPU and GPU
- **Familiar NumPy-like API**: Easy to learn for those familiar with NumPy or PyTorch
- **Composable function transformations**: Including automatic differentiation
- **Dynamic graph construction**: No need to define static computation graphs

MLX Swift brings all these capabilities to Swift, enabling native iOS and macOS applications with on-device machine learning.

### Key Benefits

- **Performance**: Optimized for Apple Silicon M-series chips
- **Memory Efficiency**: Uses unified memory, avoiding costly data transfers
- **Swift Integration**: First-class Swift API with type safety and modern Swift features
- **On-Device ML**: Run large language models and other ML workloads locally
- **SwiftUI Compatible**: Easy integration with SwiftUI for building user interfaces

---

## 2. CORE CONCEPTS AND ARCHITECTURE

### Lazy Evaluation

MLX uses lazy evaluation to optimize performance. Operations on arrays are not executed immediately; instead, they're recorded in a computation graph. The graph is compiled and executed only when the results are needed (e.g., when you call `eval()`).

```swift
let a = MLXArray(0 ..< 100)
let b = a * 2  // Not yet computed
let c = b + 10  // Still not computed
let result = c.eval()  // Now the entire graph is compiled and executed
```

### Arrays and Devices

MLX arrays can exist on different devices (CPU or GPU). Operations automatically handle device placement, but you can explicitly control this:

```swift
// Create array on GPU
let gpuArray = MLXArray([1, 2, 3], device: .gpu)

// Create array on CPU
let cpuArray = MLXArray([4, 5, 6], device: .cpu)
```

### Broadcasting

Like NumPy, MLX supports broadcasting - automatic expansion of arrays with different shapes during arithmetic operations:

```swift
let a = MLXArray([1, 2, 3])  // shape [3]
let b = MLXArray([[1], [2], [3]])  // shape [3, 1]
let c = a + b  // Result has shape [3, 3] via broadcasting
```

---

## 3. MLXARRAY - THE FUNDAMENTAL DATA STRUCTURE

`MLXArray` is the core data structure in MLX Swift, similar to NumPy's ndarray or PyTorch's Tensor.

### Creating MLXArray

```swift
// From Swift arrays
let arr1 = MLXArray([1, 2, 3, 4])

// From ranges
let arr2 = MLXArray(0 ..< 100)

// With specific shape
let arr3 = MLXArray(0 ..< 12, [3, 4])  // 3x4 matrix

// Zeros, ones, and other constructors
let zeros = MLXArray.zeros([10, 10])
let ones = MLXArray.ones([5, 5])
let random = MLXArray.random(0 ..< 1, [100, 100])

// From scalar
let scalar = MLXArray(3.14)

// With specific dtype
let floats = MLXArray([1, 2, 3], dtype: .float32)
let ints = MLXArray([1.5, 2.7, 3.9], dtype: .int32)
```

### Array Properties

```swift
let arr = MLXArray(0 ..< 24, [2, 3, 4])

arr.shape       // [2, 3, 4]
arr.ndim        // 3
arr.size        // 24
arr.dtype       // .int32 or .float32, etc.
arr.device      // .cpu or .gpu
```

### Indexing and Slicing

```swift
let matrix = MLXArray(0 ..< 20, [4, 5])

// Single element
let element = matrix[2, 3]

// Slicing rows
let firstRow = matrix[0]
let lastRow = matrix[-1]

// Slicing columns
let firstCol = matrix[0..., 0]

// Range slicing
let subMatrix = matrix[1..<3, 2..<4]

// Strided slicing
let everyOther = matrix[0..., .stride(by: 2)]
```

### Reshaping and Transposing

```swift
let arr = MLXArray(0 ..< 12)

// Reshape
let reshaped = arr.reshaped([3, 4])
let flattened = reshaped.flattened()

// Transpose
let matrix = MLXArray(0 ..< 6, [2, 3])
let transposed = matrix.T
let permuted = matrix.transposed(axes: [1, 0])

// Squeeze and expand dims
let squeezed = matrix.squeezed()
let expanded = matrix.expandedDimensions(axis: 0)
```

---

## 4. OPERATIONS AND TRANSFORMATIONS

### Arithmetic Operations

```swift
let a = MLXArray([1, 2, 3, 4])
let b = MLXArray([5, 6, 7, 8])

// Element-wise operations
let sum = a + b
let diff = a - b
let product = a * b
let quotient = a / b
let power = pow(a, b)
let sqrt = sqrt(a)

// Scalar operations
let scaled = a * 2.0
let offset = a + 10

// In-place operations
var c = a
c += b
c *= 2
```

### Mathematical Functions

```swift
let x = MLXArray([0, 0.5, 1.0, 1.5, 2.0])

// Trigonometric
let sinValues = sin(x)
let cosValues = cos(x)
let tanValues = tan(x)

// Exponential and logarithmic
let expValues = exp(x)
let logValues = log(x + 1)  // Avoid log(0)
let log10Values = log10(x + 1)

// Rounding
let rounded = round(x)
let ceiled = ceil(x)
let floored = floor(x)

// Absolute value and sign
let abs = abs(x)
let sign = sign(x)

// Clipping
let clipped = clip(x, min: 0.5, max: 1.5)
```

### Reduction Operations

```swift
let matrix = MLXArray(1 ... 12, [3, 4])

// Sum
let totalSum = matrix.sum()  // Sum all elements
let rowSum = matrix.sum(axis: 1)  // Sum along rows
let colSum = matrix.sum(axis: 0)  // Sum along columns

// Mean
let mean = matrix.mean()
let rowMean = matrix.mean(axis: 1)

// Min and Max
let minimum = matrix.min()
let maximum = matrix.max()
let argmin = matrix.argMin()  // Index of minimum
let argmax = matrix.argMax()  // Index of maximum

// Standard deviation and variance
let std = matrix.std()
let variance = matrix.variance()

// Product
let prod = matrix.product()

// All and Any (for boolean arrays)
let condition = matrix > 5
let any = condition.any()
let all = condition.all()
```

### Linear Algebra

```swift
// Matrix multiplication
let A = MLXArray.random(0 ..< 1, [3, 4])
let B = MLXArray.random(0 ..< 1, [4, 5])
let C = matmul(A, B)  // Result is [3, 5]

// Dot product (for vectors)
let v1 = MLXArray([1, 2, 3])
let v2 = MLXArray([4, 5, 6])
let dotProduct = dot(v1, v2)

// Outer product
let outer = outerProduct(v1, v2)

// Batched matrix multiplication
let batch1 = MLXArray.random(0 ..< 1, [10, 3, 4])
let batch2 = MLXArray.random(0 ..< 1, [10, 4, 5])
let batchResult = matmul(batch1, batch2)  // [10, 3, 5]
```

### Comparison and Logical Operations

```swift
let a = MLXArray([1, 2, 3, 4, 5])
let b = MLXArray([3, 3, 3, 3, 3])

// Comparison operators
let equal = a == b
let notEqual = a != b
let less = a < b
let lessOrEqual = a <= b
let greater = a > b
let greaterOrEqual = a >= b

// Logical operations
let and = equal && notEqual
let or = equal || notEqual
let not = !equal

// Where (conditional selection)
let selected = MLXArray.where(a > 3, a, b)  // Select from a or b based on condition
```

### Concatenation and Stacking

```swift
let a = MLXArray([1, 2, 3])
let b = MLXArray([4, 5, 6])

// Concatenate
let concatenated = MLXArray.concatenated([a, b], axis: 0)  // [1, 2, 3, 4, 5, 6]

// Stack
let stacked = MLXArray.stacked([a, b], axis: 0)  // [[1, 2, 3], [4, 5, 6]]

// Split
let parts = concatenated.split(indices: [3], axis: 0)  // Split at index 3
```

---

## 5. NEURAL NETWORK MODULE (MLX.NN)

MLX.NN provides building blocks for neural networks, similar to PyTorch's nn.Module.

### Module Base Class

All neural network layers inherit from `Module`:

```swift
class Module {
    // Parameters are stored as MLXArray
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // Forward pass implementation
        fatalError("Must override")
    }
    
    // Get all trainable parameters
    func parameters() -> [String: MLXArray] {
        // Returns dictionary of parameter name -> array
    }
    
    // Update parameters
    func update(parameters: [String: MLXArray]) {
        // Update module parameters
    }
    
    // Training mode
    var training: Bool = true
}
```

### Common Layers

#### Linear Layer

```swift
class Linear: Module {
    let weight: MLXArray
    let bias: MLXArray?
    
    init(inputDims: Int, outputDims: Int, bias: Bool = true) {
        // Initialize weights and biases
        self.weight = MLXArray.random(-1 ..< 1, [outputDims, inputDims])
        if bias {
            self.bias = MLXArray.zeros([outputDims])
        } else {
            self.bias = nil
        }
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = matmul(x, weight.T)
        if let bias = bias {
            output = output + bias
        }
        return output
    }
}

// Usage
let linear = Linear(inputDims: 128, outputDims: 64)
let input = MLXArray.random(0 ..< 1, [32, 128])  // Batch of 32
let output = linear(input)  // [32, 64]
```

#### Embedding Layer

```swift
class Embedding: Module {
    let weight: MLXArray
    
    init(vocabularySize: Int, dimensions: Int) {
        self.weight = MLXArray.random(-0.1 ..< 0.1, [vocabularySize, dimensions])
    }
    
    override func callAsFunction(_ indices: MLXArray) -> MLXArray {
        // Look up embeddings for given indices
        return weight[indices]
    }
}

// Usage
let embedding = Embedding(vocabularySize: 10000, dimensions: 256)
let tokens = MLXArray([1, 42, 100, 256])
let embeddings = embedding(tokens)  // [4, 256]
```

#### Convolutional Layers

```swift
class Conv2d: Module {
    let weight: MLXArray
    let bias: MLXArray?
    let stride: (Int, Int)
    let padding: (Int, Int)
    
    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int),
        stride: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0),
        bias: Bool = true
    ) {
        self.weight = MLXArray.random(
            -0.1 ..< 0.1,
            [outChannels, inChannels, kernelSize.0, kernelSize.1]
        )
        if bias {
            self.bias = MLXArray.zeros([outChannels])
        } else {
            self.bias = nil
        }
        self.stride = stride
        self.padding = padding
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Convolution operation
        var output = conv2d(x, weight, stride: stride, padding: padding)
        if let bias = bias {
            output = output + bias.reshaped([1, -1, 1, 1])
        }
        return output
    }
}
```

#### Normalization Layers

```swift
class LayerNorm: Module {
    let weight: MLXArray
    let bias: MLXArray
    let eps: Float
    
    init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.bias = MLXArray.zeros([dimensions])
        self.eps = eps
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normalized = (x - mean) / sqrt(variance + eps)
        return normalized * weight + bias
    }
}

class RMSNorm: Module {
    let weight: MLXArray
    let eps: Float
    
    init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = sqrt(x.square().mean(axis: -1, keepDims: true) + eps)
        return (x / rms) * weight
    }
}
```

#### Activation Functions

```swift
// ReLU
func relu(_ x: MLXArray) -> MLXArray {
    return maximum(x, 0)
}

// GELU
func gelu(_ x: MLXArray) -> MLXArray {
    return x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
}

// SiLU (Swish)
func silu(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(x)
}

// Sigmoid
func sigmoid(_ x: MLXArray) -> MLXArray {
    return 1.0 / (1.0 + exp(-x))
}

// Tanh
func tanh(_ x: MLXArray) -> MLXArray {
    return MLX.tanh(x)
}

// Softmax
func softmax(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let maxVal = x.max(axis: axis, keepDims: true)
    let expX = exp(x - maxVal)
    return expX / expX.sum(axis: axis, keepDims: true)
}

// Log Softmax
func logSoftmax(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let maxVal = x.max(axis: axis, keepDims: true)
    let shifted = x - maxVal
    return shifted - log(exp(shifted).sum(axis: axis, keepDims: true))
}
```

#### Dropout

```swift
class Dropout: Module {
    let probability: Float
    
    init(probability: Float = 0.5) {
        self.probability = probability
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard training else { return x }
        
        let mask = MLXArray.random(0 ..< 1, x.shape) > probability
        return x * mask / (1 - probability)
    }
}
```

### Sequential Container

```swift
class Sequential: Module {
    let layers: [Module]
    
    init(_ layers: Module...) {
        self.layers = layers
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = x
        for layer in layers {
            output = layer(output)
        }
        return output
    }
}

// Usage
let model = Sequential(
    Linear(inputDims: 784, outputDims: 256),
    ReLU(),
    Dropout(probability: 0.5),
    Linear(inputDims: 256, outputDims: 128),
    ReLU(),
    Linear(inputDims: 128, outputDims: 10)
)
```

---

## 6. LANGUAGE MODEL SUPPORT

MLX Swift provides specialized support for running large language models efficiently.

### Transformer Components

#### Multi-Head Attention

```swift
class MultiHeadAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    
    let queryProj: Linear
    let keyProj: Linear
    let valueProj: Linear
    let outProj: Linear
    
    init(dimensions: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dimensions / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        
        self.queryProj = Linear(inputDims: dimensions, outputDims: dimensions)
        self.keyProj = Linear(inputDims: dimensions, outputDims: dimensions)
        self.valueProj = Linear(inputDims: dimensions, outputDims: dimensions)
        self.outProj = Linear(inputDims: dimensions, outputDims: dimensions)
    }
    
    override func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0]  // Batch size
        let L = x.shape[1]  // Sequence length
        
        // Project to Q, K, V
        var queries = queryProj(x).reshaped([B, L, numHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        var keys = keyProj(x).reshaped([B, L, numHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        var values = valueProj(x).reshaped([B, L, numHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        
        // Attention scores
        var scores = matmul(queries, keys.transposed(axes: [0, 1, 3, 2])) * scale
        
        // Apply mask if provided
        if let mask = mask {
            scores = scores + mask
        }
        
        // Softmax and apply to values
        let attn = softmax(scores, axis: -1)
        var output = matmul(attn, values)
        
        // Reshape and project
        output = output.transposed(axes: [0, 2, 1, 3]).reshaped([B, L, -1])
        return outProj(output)
    }
}
```

#### Transformer Block

```swift
class TransformerBlock: Module {
    let attention: MultiHeadAttention
    let norm1: LayerNorm
    let norm2: LayerNorm
    let mlp: Sequential
    
    init(dimensions: Int, numHeads: Int, mlpDimensions: Int) {
        self.attention = MultiHeadAttention(dimensions: dimensions, numHeads: numHeads)
        self.norm1 = LayerNorm(dimensions: dimensions)
        self.norm2 = LayerNorm(dimensions: dimensions)
        
        self.mlp = Sequential(
            Linear(inputDims: dimensions, outputDims: mlpDimensions),
            GELU(),
            Linear(inputDims: mlpDimensions, outputDims: dimensions)
        )
    }
    
    override func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Self-attention with residual
        var h = x + attention(norm1(x), mask: mask)
        
        // MLP with residual
        h = h + mlp(norm2(h))
        
        return h
    }
}
```

### KV Cache for Fast Generation

```swift
class KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    let maxLength: Int
    
    init(maxLength: Int = 2048) {
        self.maxLength = maxLength
    }
    
    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            // Concatenate with existing cache
            keys = MLXArray.concatenated([existingKeys, newKeys], axis: 2)
            values = MLXArray.concatenated([existingValues, newValues], axis: 2)
        } else {
            // Initialize cache
            keys = newKeys
            values = newValues
        }
        
        return (keys!, values!)
    }
    
    func reset() {
        keys = nil
        values = nil
    }
}
```

### Text Generation

```swift
func generateText(
    model: Module,
    tokenizer: Tokenizer,
    prompt: String,
    maxTokens: Int = 100,
    temperature: Float = 0.7,
    topP: Float = 0.9
) -> String {
    // Tokenize prompt
    var tokens = tokenizer.encode(prompt)
    
    // Generate tokens
    for _ in 0 ..< maxTokens {
        // Get logits from model
        let input = MLXArray(tokens)
        let logits = model(input.reshaped([1, -1]))
        
        // Sample next token
        let nextToken = sample(
            logits: logits[0, -1],
            temperature: temperature,
            topP: topP
        )
        
        tokens.append(nextToken)
        
        // Check for end of sequence
        if nextToken == tokenizer.eosToken {
            break
        }
    }
    
    return tokenizer.decode(tokens)
}

func sample(logits: MLXArray, temperature: Float, topP: Float) -> Int {
    // Apply temperature
    let scaledLogits = logits / temperature
    
    // Convert to probabilities
    let probs = softmax(scaledLogits)
    
    // Top-p (nucleus) sampling
    let sortedProbs = probs.sorted(descending: true)
    let cumulativeProbs = sortedProbs.cumsum()
    
    let mask = cumulativeProbs <= topP
    let filteredProbs = MLXArray.where(mask, sortedProbs, 0)
    
    // Sample from filtered distribution
    let normalizedProbs = filteredProbs / filteredProbs.sum()
    return categorical(normalizedProbs)
}
```

### Streaming Generation

```swift
func streamingGenerate(
    model: Module,
    tokenizer: Tokenizer,
    prompt: String,
    maxTokens: Int = 100,
    onToken: (String) -> Void
) async {
    var tokens = tokenizer.encode(prompt)
    
    for _ in 0 ..< maxTokens {
        let input = MLXArray(tokens)
        let logits = model(input.reshaped([1, -1]))
        
        let nextToken = sample(logits: logits[0, -1], temperature: 0.7, topP: 0.9)
        tokens.append(nextToken)
        
        // Stream token to callback
        let tokenText = tokenizer.decode([nextToken])
        onToken(tokenText)
        
        if nextToken == tokenizer.eosToken {
            break
        }
        
        // Allow UI updates
        await Task.yield()
    }
}
```

---

## 7. MEMORY MANAGEMENT AND PERFORMANCE

### Evaluation and Materialization

MLX uses lazy evaluation. To force computation:

```swift
let arr = MLXArray(0 ..< 1000) * 2 + 5

// Force evaluation
arr.eval()

// Or access data (implicitly evaluates)
let item = arr.item()  // For scalars
let data = arr.asArray([Int].self)  // Convert to Swift array
```

### Memory Efficiency

```swift
// Free memory from arrays no longer needed
var largeArray: MLXArray? = MLXArray.random(0 ..< 1, [10000, 10000])
// ... use array ...
largeArray = nil  // Release memory

// Explicit memory management
MLX.clearCache()  // Clear MLX's internal cache
```

### Performance Monitoring

```swift
import Foundation

func measurePerformance<T>(_ operation: () -> T) -> (result: T, duration: TimeInterval) {
    let start = Date()
    let result = operation()
    let duration = Date().timeIntervalSince(start)
    return (result, duration)
}

// Usage
let (result, duration) = measurePerformance {
    let arr = MLXArray.random(0 ..< 1, [1000, 1000])
    let product = matmul(arr, arr)
    return product.eval()
}

print("Operation took \(duration) seconds")
```

### Tokens Per Second Tracking

```swift
class PerformanceTracker {
    private var startTime: Date?
    private var tokenCount: Int = 0
    
    func start() {
        startTime = Date()
        tokenCount = 0
    }
    
    func recordToken() {
        tokenCount += 1
    }
    
    func tokensPerSecond() -> Double {
        guard let start = startTime else { return 0 }
        let elapsed = Date().timeIntervalSince(start)
        return Double(tokenCount) / elapsed
    }
    
    func reset() {
        startTime = nil
        tokenCount = 0
    }
}
```

### Batching for Efficiency

```swift
// Process multiple inputs together
let batchSize = 32
let inputs = MLXArray.random(0 ..< 1, [batchSize, 128])

// Single forward pass for entire batch
let outputs = model(inputs)  // More efficient than processing one-by-one
```

---

## 8. COMPLETE API REFERENCE

### MLXArray Static Methods

```swift
// Creation methods
MLXArray.zeros(_ shape: [Int], dtype: DType = .float32, device: Device = .default)
MLXArray.ones(_ shape: [Int], dtype: DType = .float32, device: Device = .default)
MLXArray.full(_ shape: [Int], value: Float, dtype: DType = .float32)
MLXArray.arange(start: Int = 0, end: Int, step: Int = 1, dtype: DType = .int32)
MLXArray.linspace(start: Float, end: Float, count: Int, dtype: DType = .float32)
MLXArray.random(_ range: Range<Float>, _ shape: [Int], dtype: DType = .float32)
MLXArray.randomNormal(mean: Float = 0, std: Float = 1, _ shape: [Int])
MLXArray.eye(_ n: Int, m: Int? = nil, k: Int = 0, dtype: DType = .float32)

// Concatenation and stacking
MLXArray.concatenated(_ arrays: [MLXArray], axis: Int = 0)
MLXArray.stacked(_ arrays: [MLXArray], axis: Int = 0)

// Conditional
MLXArray.where(_ condition: MLXArray, _ x: MLXArray, _ y: MLXArray)
```

### MLXArray Instance Methods

```swift
// Shape manipulation
func reshaped(_ shape: [Int]) -> MLXArray
func flattened() -> MLXArray
func squeezed(axis: Int? = nil) -> MLXArray
func expandedDimensions(axis: Int) -> MLXArray
func transposed(axes: [Int]? = nil) -> MLXArray

// Arithmetic
func sum(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func mean(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func min(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func max(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func product(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func cumsum(axis: Int = -1) -> MLXArray

// Statistical
func variance(axis: Int? = nil, keepDims: Bool = false, ddof: Int = 0) -> MLXArray
func std(axis: Int? = nil, keepDims: Bool = false, ddof: Int = 0) -> MLXArray

// Searching
func argMin(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func argMax(axis: Int? = nil, keepDims: Bool = false) -> MLXArray

// Logic
func any(axis: Int? = nil, keepDims: Bool = false) -> MLXArray
func all(axis: Int? = nil, keepDims: Bool = false) -> MLXArray

// Evaluation
func eval() -> MLXArray
func item() -> Scalar  // For single-element arrays
func asArray<T>(_ type: T.Type) -> [T]  // Convert to Swift array

// Type conversion
func asType(_ dtype: DType) -> MLXArray
```

### Global Functions

```swift
// Math operations
func sqrt(_ x: MLXArray) -> MLXArray
func exp(_ x: MLXArray) -> MLXArray
func log(_ x: MLXArray) -> MLXArray
func log2(_ x: MLXArray) -> MLXArray
func log10(_ x: MLXArray) -> MLXArray
func sin(_ x: MLXArray) -> MLXArray
func cos(_ x: MLXArray) -> MLXArray
func tan(_ x: MLXArray) -> MLXArray
func abs(_ x: MLXArray) -> MLXArray
func sign(_ x: MLXArray) -> MLXArray
func square(_ x: MLXArray) -> MLXArray
func pow(_ x: MLXArray, _ y: MLXArray) -> MLXArray
func minimum(_ x: MLXArray, _ y: MLXArray) -> MLXArray
func maximum(_ x: MLXArray, _ y: MLXArray) -> MLXArray
func clip(_ x: MLXArray, min: Float, max: Float) -> MLXArray

// Linear algebra
func matmul(_ a: MLXArray, _ b: MLXArray) -> MLXArray
func dot(_ a: MLXArray, _ b: MLXArray) -> MLXArray
func outerProduct(_ a: MLXArray, _ b: MLXArray) -> MLXArray

// Neural network operations
func conv2d(_ input: MLXArray, _ weight: MLXArray, stride: (Int, Int), padding: (Int, Int)) -> MLXArray
func maxPool2d(_ input: MLXArray, kernelSize: (Int, Int), stride: (Int, Int)) -> MLXArray
func softmax(_ x: MLXArray, axis: Int) -> MLXArray
func logSoftmax(_ x: MLXArray, axis: Int) -> MLXArray
func relu(_ x: MLXArray) -> MLXArray
func gelu(_ x: MLXArray) -> MLXArray
func silu(_ x: MLXArray) -> MLXArray
func sigmoid(_ x: MLXArray) -> MLXArray
func tanh(_ x: MLXArray) -> MLXArray
```

### Data Types

```swift
enum DType {
    case bool
    case uint8
    case uint16
    case uint32
    case uint64
    case int8
    case int16
    case int32
    case int64
    case float16
    case float32
    case bfloat16
    case complex64
}
```

### Devices

```swift
enum Device {
    case cpu
    case gpu
    case `default`  // Let MLX choose
    
    static var current: Device { get }
}
```

---

## 9. CODE EXAMPLES

### Example 1: Simple Neural Network

```swift
class SimpleClassifier: Module {
    let layer1: Linear
    let layer2: Linear
    let layer3: Linear
    let dropout: Dropout
    
    init(inputSize: Int, hiddenSize: Int, numClasses: Int) {
        self.layer1 = Linear(inputDims: inputSize, outputDims: hiddenSize)
        self.layer2 = Linear(inputDims: hiddenSize, outputDims: hiddenSize)
        self.layer3 = Linear(inputDims: hiddenSize, outputDims: numClasses)
        self.dropout = Dropout(probability: 0.5)
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(layer1(x))
        h = dropout(h)
        h = relu(layer2(h))
        h = dropout(h)
        return layer3(h)
    }
}

// Training
let model = SimpleClassifier(inputSize: 784, hiddenSize: 256, numClasses: 10)
let optimizer = SGD(learningRate: 0.01)

for epoch in 0 ..< 10 {
    for (batch, labels) in dataLoader {
        // Forward pass
        let logits = model(batch)
        let loss = crossEntropyLoss(logits: logits, labels: labels)
        
        // Backward pass
        let gradients = grad(loss, model.parameters())
        
        // Update weights
        optimizer.update(model, gradients: gradients)
    }
}
```

### Example 2: Image Processing

```swift
func processImage(_ image: MLXArray) -> MLXArray {
    // Input: [H, W, C] image
    // Normalize to [-1, 1]
    var processed = (image / 127.5) - 1.0
    
    // Apply Gaussian blur (simple box filter approximation)
    let kernel = MLXArray.ones([3, 3, 1, 1]) / 9.0
    processed = conv2d(
        processed.expandedDimensions(axis: 0),
        kernel,
        stride: (1, 1),
        padding: (1, 1)
    )
    
    // Edge detection
    let sobelX = MLXArray([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).reshaped([3, 3, 1, 1])
    
    let edges = conv2d(processed, sobelX, stride: (1, 1), padding: (1, 1))
    
    return edges.squeezed(axis: 0)
}
```

### Example 3: Text Embedding Similarity

```swift
func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
    let dotProduct = (a * b).sum()
    let normA = sqrt((a * a).sum())
    let normB = sqrt((b * b).sum())
    return (dotProduct / (normA * normB)).item()
}

let embedding = Embedding(vocabularySize: 10000, dimensions: 512)

let sentence1 = MLXArray([42, 123, 456, 789])
let sentence2 = MLXArray([42, 125, 458, 791])

let emb1 = embedding(sentence1).mean(axis: 0)
let emb2 = embedding(sentence2).mean(axis: 0)

let similarity = cosineSimilarity(emb1, emb2)
print("Similarity: \(similarity)")
```

### Example 4: Attention Mechanism

```swift
func scaledDotProductAttention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    mask: MLXArray? = nil
) -> MLXArray {
    let dk = Float(query.shape[-1])
    var scores = matmul(query, key.transposed(axes: [0, 1, 3, 2])) / sqrt(dk)
    
    if let mask = mask {
        scores = scores + mask
    }
    
    let attention = softmax(scores, axis: -1)
    return matmul(attention, value)
}
```

### Example 5: LLM Inference with Performance Tracking

```swift
class LLMInference {
    let model: Module
    let tokenizer: Tokenizer
    let performanceTracker = PerformanceTracker()
    
    init(model: Module, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }
    
    func generate(prompt: String, maxTokens: Int = 100) -> (text: String, tokensPerSecond: Double) {
        performanceTracker.start()
        
        var tokens = tokenizer.encode(prompt)
        var generatedText = prompt
        
        for _ in 0 ..< maxTokens {
            let input = MLXArray(tokens).reshaped([1, -1])
            let logits = model(input)
            
            let nextToken = sample(logits: logits[0, -1], temperature: 0.7, topP: 0.9)
            tokens.append(nextToken)
            
            let tokenText = tokenizer.decode([nextToken])
            generatedText += tokenText
            
            performanceTracker.recordToken()
            
            if nextToken == tokenizer.eosToken {
                break
            }
        }
        
        let tps = performanceTracker.tokensPerSecond()
        return (generatedText, tps)
    }
}
```

---

# YOUR TASK: Build a Command-Line LLM Chat Tool

  Using the MLX Swift documentation above, create a **single-file command-line Swift program** that demonstrates
  core MLX concepts for LLM inference.

  ## Requirements

  Create a file called `mlx-chat.swift` that implements a simple but complete command-line chat interface with an
  LLM.

  ### 1. Model Loading

  **Load a quantized model:**
  - Use `loadModel(id:)` to load "mlx-community/Qwen2-0.5B-Instruct-4bit" (small, fast model)
  - Show loading progress with simple text updates
  - Handle loading errors gracefully

  **In your comments, explain:**
  - Why quantized models (4-bit) are used for on-device inference
  - How lazy evaluation affects model loading
  - The role of unified memory in model weight storage

  ### 2. Chat Session with Streaming

  **Implement a REPL (Read-Eval-Print Loop):**
  - Use `ChatSession` for conversation management
  - Stream responses token-by-token to stdout using `streamResponse(to:)`
  - Support multi-turn conversations (context preservation)
  - Allow user to type "exit" or "quit" to end
  - Clear command ("clear") to reset conversation

  **In your comments, explain:**
  - How KV cache enables efficient multi-turn chat
  - Why streaming is important for user experience
  - The difference between eager and lazy evaluation during generation

  ### 3. Performance Metrics

  **Track and display after each response:**
  - Tokens per second: calculate from generation time
  - Time to first token (TTFT)
  - Total tokens in response
  - Total generation time
  - GPU memory usage (use `GPU.snapshot()`)

  **Print metrics in a clean format:**
  ─────────────────────────────
  📊 Metrics
  ─────────────────────────────
  ⚡ Speed: 42.5 tok/s
  ⏱️  First token: 156ms
  📝 Tokens: 85
  ⏰ Total time: 2.0s
  💾 GPU memory: 1.2 GB
  ─────────────────────────────

  **In your comments, explain:**
  - How unified memory benefits performance monitoring
  - Why GPU memory doesn't need to be copied to CPU for metrics
  - The relationship between memory usage and KV cache size

  ### 4. Generation Parameters

  **Support command-line arguments:**
  ```bash
  swift mlx-chat.swift --temperature 0.7 --max-tokens 500

  Parameters to support:
  - --temperature (default: 0.6)
  - --max-tokens (default: 200)
  - --model (default: "mlx-community/Qwen2-0.5B-Instruct-4bit")

  5. Code Structure

  Single file structure (~200-300 lines):

  import Foundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - Configuration
  struct ChatConfig {
      let modelId: String
      let temperature: Float
      let maxTokens: Int

      static func parseArguments() -> ChatConfig {
          // Parse CommandLine.arguments
      }
  }

  // MARK: - Metrics Tracking
  struct GenerationMetrics {
      var tokensPerSecond: Double
      var timeToFirstToken: TimeInterval
      var totalTokens: Int
      var totalTime: TimeInterval
      var gpuMemoryMB: Double

      func display() {
          // Pretty print metrics
      }
  }

  // MARK: - Main Chat Loop
  @main
  struct MLXChat {
      static func main() async throws {
          // 1. Parse arguments
          // 2. Load model with progress
          // 3. Create ChatSession
          // 4. Enter REPL loop
          // 5. Handle streaming responses
          // 6. Track and display metrics
      }
  }

  // MARK: - Helper Functions
  func printWelcome(config: ChatConfig) { }
  func printMetrics(_ metrics: GenerationMetrics) { }
  func readInput(prompt: String) -> String? { }

  6. Detailed Comments Required

  Add explanatory comments for these MLX-specific concepts:

  Lazy Evaluation:
  // MLX uses lazy evaluation - the model weights aren't actually loaded
  // into memory until we call eval() or perform an operation that requires
  // materialization. This allows efficient weight updates and transformations
  // before any computation happens.
  let model = try await loadModel(id: config.modelId)

  KV Cache:
  // ChatSession maintains a KV (key-value) cache across conversation turns.
  // Instead of reprocessing all previous tokens on each turn, the cache
  // stores attention keys and values, enabling O(1) context instead of O(n).
  // Trade-off: memory usage grows linearly with conversation length.
  let session = ChatSession(model, generateParameters: params)

  Unified Memory:
  // Apple Silicon's unified memory means GPU and CPU share the same physical
  // memory. We can access model weights and activations from both CPU (for
  // tokenization) and GPU (for inference) without copying. This is why
  // GPU.snapshot() can report memory instantly without data transfers.
  let snapshot = GPU.snapshot()
  let memoryMB = Double(snapshot.activeMemory) / 1024 / 1024

  Streaming Generation:
  // Streaming allows displaying tokens as they're generated rather than
  // waiting for the complete response. Each iteration evaluates just enough
  // of the computation graph to produce the next token. This creates a
  // responsive user experience.
  for try await token in session.streamResponse(to: prompt) {
      print(token, terminator: "")
      fflush(stdout)  // Force immediate display
  }

  Example Session

  $ swift mlx-chat.swift --temperature 0.7 --max-tokens 300

  🤖 MLX Chat - Loading model...
  📦 Model: mlx-community/Qwen2-0.5B-Instruct-4bit
  ✅ Ready! Type 'exit' to quit, 'clear' to reset conversation.

  You: What is Swift?
