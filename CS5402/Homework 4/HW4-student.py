import numpy as np
# not allowed to import any other libraries.

class CustomConv2D:
    def __init__(self, kernel):
        """
        kernel: 2D numpy array (kH x kW)
        """
        self.kernel = kernel
        self.kH, self.kW = kernel.shape

    
    def forward(self, input):
        """
        input: 2D numpy array (H x W)

        Returns:
            output: 2D numpy array
        """
        H, W = input.shape
        # TODO:
        # Perform convolution:
        # 1. Extract patch
        # 2. Element-wise multiply with kernel
        # 3. Sum result

        return output
    





if __name__ == "__main__":
    input = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [0, 1, 3, 2, 1],
        [1, 0, 2, 1, 0]
    ])

    kernel = np.array([
        [1, 0],
        [0, -1]
    ])

    conv = CustomConv2D(kernel)
    output = conv.forward(input)

    print("Output:")
    print(output)

    # ExpectedOutput:
    #     [[ 0.  0. -1.]
    #     [-2.  0.  3.]
    #     [ 0. -1.  2.]]