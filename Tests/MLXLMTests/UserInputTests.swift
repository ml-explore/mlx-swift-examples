import Foundation
import XCTest
import MLXLMCommon
import MLX

public class MLXLMCommonTests: XCTestCase {
    
    public func testExample() {
        let x = UserInput(prompt: "foo")
        print(x)
        
        let a = MLXArray(10)
        print(a + 1)
    }
    
}
