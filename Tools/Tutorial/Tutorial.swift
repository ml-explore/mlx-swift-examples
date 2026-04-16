// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// mlx-swift tutorial based on:
/// https://github.com/ml-explore/mlx/blob/main/examples/cpp/tutorial.cpp
@main
struct Tutorial {

    static func scalarBasics() {
        // create a scalar array
        let x = MLXArray(1.0)

        // the datatype is .float32
        let dtype = x.dtype
        assert(dtype == .float32)

        // get the value
        let s = x.item(Float.self)
        assert(s == 1.0)

        // reading the value with a different type is a fatal error
        // let i = x.item(Int.self)

        // scalars have a size of 1
        let size = x.size
        assert(size == 1)

        // scalars have 0 dimensions
        let ndim = x.ndim
        assert(ndim == 0)

        // scalar shapes are empty arrays
        let shape = x.shape
        assert(shape == [])
    }

    static func arrayBasics() {
        // make a multidimensional array.
        //
        // Note: the argument is a [Double] array literal, which is not
        // a supported type, but we can explicitly convert it to [Float]
        // when we create the MLXArray.
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0], [2, 2])

        // mlx is row-major by default so the first row of this array
        // is [1.0, 2.0] and the second row is [3.0, 4.0]
        print(x[0])
        print(x[1])

        // make an array of shape [2, 2] filled with ones
        let y = MLXArray.ones([2, 2])

        // pointwise add x and y
        let z = x + y

        // mlx is lazy by default. At this point `z` only
        // has a shape and a type but no actual data
        assert(z.dtype == .float32)
        assert(z.shape == [2, 2])

        // To actually run the computation you must evaluate `z`.
        // Under the hood, mlx records operations in a graph.
        // The variable `z` is a node in the graph which points to its operation
        // and inputs. When `eval` is called on an array (or arrays), the array and
        // all of its dependencies are recursively evaluated to produce the result.
        // Once an array is evaluated, it has data and is detached from its inputs.

        // Note: this is being called for demonstration purposes -- all reads
        // ensure the array is evaluated.
        z.eval()

        // this implicitly evaluates z before converting to a description
        print(z)
    }

    static func automaticDifferentiation() {
        func fn(_ x: MLXArray) -> MLXArray {
            x.square()
        }

        let gradFn = grad(fn)

        let x = MLXArray(1.5)
        let dfdx = gradFn(x)
        print(dfdx)

        assert(dfdx.item() == Float(2 * 1.5))

        let df2dx2 = grad(grad(fn))(x)
        print(df2dx2)

        assert(df2dx2.item() == Float(2))
    }

    static func main() {
        scalarBasics()
        arrayBasics()
        automaticDifferentiation()
    }
}
