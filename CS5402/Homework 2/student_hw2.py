import math
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter, defaultdict
# Do not import any other libraries. You can use built-in functions and the above imports only.


###################### Task-1 ################################################

def entropy(labels):
    """
    Compute entropy of a list of class labels.
    """
    counts = Counter(labels)
    total_labels = sum(counts.values())

    total_entropy = 0
    for count in counts.values():
        prob = count / total_labels
        total_entropy += (prob * math.log2(prob))
    
    return -total_entropy


def information_gain(dataset):
    """
    Input:
        dataset: list of lists
                 last column is label
    Output:
        list of information gain for each feature
    """
    num_features = len(dataset[0]) - 1
    labels = [row[-1] for row in dataset]

    total_entropy = entropy(labels)
    ig_list = []

    for feature_idx in range(num_features):
        # Read the dataset and group all labels for each feature
        # (Basically splitting the dataset on each feature)
        subsets = defaultdict(list)
        for row in dataset:
            feature_value = row[feature_idx]
            label = row[-1]
            subsets[feature_value].append(label)

        conditional_entropy = 0
        total_samples = len(dataset)

        # Calculate the new entropy for each split
        for subset_labels in subsets.values():
            weight = len(subset_labels) / total_samples
            conditional_entropy += weight * entropy(subset_labels)

        # Information gain
        ig = total_entropy - conditional_entropy
        ig_list.append(ig)

    return ig_list



###################### Task-2 ################################################
def perceptron_gradient_descent(X, y, w_init, b_init, lr=1.0, max_iter=100):
    """
    Parameters:
        X : list of feature vectors
        y : list of labels (-1 or +1)
        w_init : initial weight vector
        b_init : initial bias
        lr : learning rate
        max_iter : maximum iterations
        
    Returns:
        w, b
    """
    # TODO: implement perceptron learning algorithm with gradient descent 
        
    
    return w, b