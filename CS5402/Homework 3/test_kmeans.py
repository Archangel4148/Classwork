import numpy as np
from sklearn.datasets import make_blobs
from kmeans_hw3 import compute_elbow, kmeans, kmeans_with_restarts


def test_basic_run():
    """
    Test whether K-Means runs without crashing.
    """

    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

    labels, centroids, sse = kmeans(X, K=3)

    assert labels.shape[0] == X.shape[0], "Incorrect label size"
    assert centroids.shape[0] == 3, "Incorrect centroid count"
    assert isinstance(sse, float), "SSE should be a float"

    print("Basic run test passed.")


def test_restart_function():
    """
    Test multiple restart functionality.
    """

    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

    labels, centroids, sse = kmeans_with_restarts(X, K=3, n_runs=5)

    assert labels.shape[0] == X.shape[0]
    assert centroids.shape[0] == 3

    print("Restart test passed.")


def test_elbow_behavior():
    """
    Basic sanity check for elbow method behavior.
    """

    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    sse_values = []

    for K in range(1, 7):

        _, _, sse = kmeans_with_restarts(X, K, n_runs=5)

        sse_values.append(sse)

    print("SSE values:", list(map(float, sse_values)))

    assert sse_values[0] > sse_values[-1], "SSE should decrease with larger K"

    print("Elbow behavior test passed.")

    # NOTE: Added this method call to show elbow method function (+plot) on test run
    sse_values, best_K = compute_elbow(X, 15)
    print("SSE Values:", list(map(float, sse_values)))
    print("Best K:", best_K)

if __name__ == "__main__":

    test_basic_run()
    test_restart_function()
    test_elbow_behavior()

    print("\nAll tests completed.")
