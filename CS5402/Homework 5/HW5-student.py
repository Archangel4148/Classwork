import numpy as np
# do not import any other libraries.

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, use_scale):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.use_scale = use_scale

        # TODO 1: Initialize weight matrices
        # Shapes:
        # W_q, W_k, W_v: (d_model, d_model)
        self.W_q = np.random.randn(self.d_model, self.d_model)
        self.W_k = np.random.randn(self.d_model, self.d_model)
        self.W_v = np.random.randn(self.d_model, self.d_model)

    def split_heads(self, x):
        """
        Input:
            x: (batch_size, seq_len, d_model)
        Output:
            (batch_size, num_heads, seq_len, d_k)
        """
        B, N, D = x.shape

        # TODO 2:
        # Step 1: reshape to (B, N, num_heads, d_k)
        x = x.reshape(B, N, self.num_heads, self.d_k)

        # Step 2: transpose to (B, num_heads, N, d_k)
        return x.transpose(0, 2, 1, 3)  # Switch num_heads and N

    def combine_heads(self, x):
        """
        Input:
            x: (batch_size, num_heads, seq_len, d_k)
        Output:
            (batch_size, seq_len, d_model)
        """
        B, H, N, d_k = x.shape

        # TODO 3:
        # Step 1: transpose to (B, N, H, d_k)
        x = x.transpose(0, 2, 1, 3)  # Switch H and N

        # Step 2: reshape to (B, N, d_model)
        return x.reshape(B, N, self.d_model)
        

    def softmax(self, x):
        """
        Standard softmax (along last dimension)
        """
        # TODO 4:
        # exp
        exps = np.exp(x)

        # divide by sum along last axis
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def scaled_dot_product_attention(self, Q, K, V, use_scale=True):
        """
        Q, K, V: (B, num_heads, seq_len, d_k)
        """
        # TODO 5: compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))

        # TODO 6: scale
        if use_scale:  # Flag to run experiments for report
            scores = scores / np.sqrt(self.d_k)

        # TODO 7: softmax
        attn = self.softmax(scores)

        # TODO 8: weighted sum
        output = np.matmul(attn, V)

        return output, attn

    def forward(self, X):
        """
        X: (batch_size, seq_len, d_model)
        """
        # TODO 9: linear projections
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)

        # TODO: print the following intermediate shapes:
        # after split:
        # attention matrix (head 0):

        # TODO 10: split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
 
        print("after split: Q shape:", Q.shape)
        print("after split: K shape:", K.shape)
        print("after split: V shape:", V.shape)

        # TODO 11: attention
        out, attn = self.scaled_dot_product_attention(Q, K, V, use_scale=self.use_scale)

        # print("attention matrix (head 0):\n", attn[0, 0])
        for h in range(attn.shape[1]):
            print(f"Head {h}:\n", attn[0, h])

        # TODO 12: combine heads
        out = self.combine_heads(out)

        return out


if __name__ == "__main__":
    np.random.seed(0)

    B, N, D = 2, 4, 8
    H = 4

    X = np.random.randn(B, N, D)

    mha = MultiHeadAttention(d_model=D, num_heads=H, use_scale=True)

    # TODO: initialize weights before running
    output = mha.forward(X)

    print("Output shape:", output.shape)  # Expected: (2, 4, 8)