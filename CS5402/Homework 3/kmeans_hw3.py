import numpy as np
# Do not import any additional libraries here. You may only use numpy and standard Python libraries.


def initialize_centroids(X, K):
    """
    Randomly initialize K centroids from the dataset.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
    K : int
        Number of clusters

    Returns
    -------
    centroids : numpy array of shape (K, n_features)
    """
    indices = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[indices]
    return centroids


def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid.

    Parameters
    ----------
    X : numpy array (n_samples, n_features)
    centroids : numpy array (K, n_features)

    Returns
    -------
    labels : numpy array (n_samples,)
        Cluster index for each data point
    """
    distances = []
    for centroid in centroids:
        d = np.linalg.norm(X - centroid, axis=1)
        distances.append(d)
    distances = np.array(distances).T
    
    labels = np.argmin(distances, axis=1)
    return labels


def update_centroids(X, labels, K):
    """
    Update centroid positions based on cluster assignments.

    Parameters
    ----------
    X : numpy array (n_samples, n_features)
    labels : numpy array (n_samples,)
    K : int

    Returns
    -------
    centroids : numpy array (K, n_features)
    """
    centroids = []
    for group in range(K):
        cluster_points = X[labels == group]
        
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            # If there aren't any points, just use a random point
            new_centroid = X[np.random.randint(0, X.shape[0])]
        centroids.append(new_centroid)
    
    return np.array(centroids)


def compute_sse(X, labels, centroids):
    """
    Compute the Sum of Squared Errors (SSE).

    Parameters
    ----------
    X : numpy array
    labels : numpy array
    centroids : numpy array

    Returns
    -------
    sse : float
    """

    sse = 0.0
    num_points = X.shape[0]
    for i in range(num_points):
        centroid = centroids[labels[i]]
        sse += np.sum((X[i] - centroid) ** 2)
    
    return sse


def kmeans(X, K, max_iter=100, tolerance=0.0001):
    """
    Basic K-Means clustering algorithm.

    Parameters
    ----------
    X : numpy array (n_samples, n_features)
    K : int
    max_iter : int

    Returns
    -------
    labels : numpy array
    centroids : numpy array
    sse : float
    """

    # Step 1: initialize centroids
    centroids = initialize_centroids(X, K)

    for _ in range(max_iter):

        # Step 2: assign clusters
        labels = assign_clusters(X, centroids)

        # Step 3: update centroids
        new_centroids = update_centroids(X, labels, K)

        # Step 4: check convergence
        if np.allclose(centroids, new_centroids, atol=tolerance):
            break

        # Update the centroids
        centroids = new_centroids

    # Step 5: compute final SSE
    sse = compute_sse(X, labels, centroids)

    return labels, centroids, sse


def kmeans_with_restarts(X, K, n_runs=10):
    """
    Run K-Means multiple times to reduce the effect of random initialization.

    Parameters
    ----------
    X : numpy array
    K : int
    n_runs : int

    Returns
    -------
    best_labels
    best_centroids
    best_sse
    """

    best_details = (None, None, np.inf)
    for _ in range(n_runs):
        labels, centroids, sse = kmeans(X, K)
        if sse < best_details[2]:
            best_details = (labels, centroids, sse)

    return best_details


def compute_elbow(X, max_K=10):
    """Use the elbow method to find the best value of K for K-means clustering"""
    k_values = list(range(1, max_K + 1))
    sse_values = []
    for K in k_values:
        # Run k-means for each K
        _, _, sse = kmeans_with_restarts(X, K)
        sse_values.append(sse)

    # Find the elbow (Farthest point from start -> end line)
    p1 = np.array([k_values[0], sse_values[0]])
    p2 = np.array([k_values[-1], sse_values[-1]])
    distances = []
    for i in range(len(k_values)):
        p = np.array([k_values[i], sse_values[i]])
        # perpendicular distance formula (from Wikipedia)
        dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(dist)

    best_K = k_values[np.argmax(distances)]

    # Display the elbow plot
    import matplotlib.pyplot as plt
    plt.title("Elbow Method Plot")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.plot(k_values, sse_values)
    plt.show()

    return sse_values, best_K
