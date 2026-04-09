import numpy as np
# not allowed to import any other libraries.

class CustomConv2D:
    def __init__(self, kernel):
        """
        kernel: 2D numpy array (kH x kW)
        """
        self.kernel = kernel
        self.kH, self.kW = kernel.shape

    
    def _get_indices_to_convolve(self, img_dim, kernel_dim):
        """Get indices to apply the kernel (with stride 1, 2, 1, 2, ...)"""
        indices = []
        current = 0
        step_count = 0
        
        # Find all indices to apply the kernel
        while current + kernel_dim <= img_dim:
            indices.append(current)
            stride = 1 if step_count % 2 == 0 else 2  # Alternate stride
            current += stride
            step_count += 1
        return indices

    def forward(self, input):
        """
        input: 2D numpy array (H x W)

        Returns:
            output: 2D numpy array
        """
        H, W = input.shape
        
        # Get all positions to apply the kernel
        y_indices = self._get_indices_to_convolve(H, self.kH)
        x_indices = self._get_indices_to_convolve(W, self.kW)
        
        # Build the output matrix
        output = np.zeros((len(y_indices), len(x_indices)))

        # Do the convolution
        for i, y in enumerate(y_indices):
            for j, x in enumerate(x_indices):
                # Get the patch, then multiply/sum
                patch = input[y : y + self.kH, x : x + self.kW]
                output[i, j] = np.sum(patch * self.kernel)
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