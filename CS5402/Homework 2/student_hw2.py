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
    w = np.array(w_init[:])
    b = b_init

    num_samples = len(X)

    for _ in range(max_iter):
        wrong = []

        # Step 1: Find the incorrect samples
        for i in range(num_samples):
            # Calculate the output and check for a mistake
            activation = w.T @ X[i] + b
            if y[i] * activation <= 0:
                wrong.append(i)
    
        # If there are no mistakes, it's done!
        if not wrong:
            break

        # Step 2: Pick the mistake with the biggest ID
        i = max(wrong)

        # Step 3: Do a gradient descent step
        for j in range(len(w)):
            w[j] -= lr * y[i] * X[i][j]
        b -= lr * y[i]

    return w, b

###################### Task-3 ################################################

# Part 1: Make the dataset
np.random.seed(0)
x_pos = np.random.randn(25, 2) + np.array([2, 2])
y_pos = np.ones(25)
x_neg = np.random.randn(25, 2) + np.array([-2, -2])
y_neg = -np.ones(25)
x = np.vstack((x_pos, x_neg))
y = np.hstack((y_pos, y_neg))

# Part 2: Perceptron Experiments
# Function to plot the boundary (from lecture notes)
def plot_boundary(w, b, x_range, style='k-', label=None):
    x_vals = np.array(x_range)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, style, label=label)

# Plot data
plt.figure(figsize=(7,7))
plt.scatter(x_pos[:,0], x_pos[:,1], color='blue', label="+1 Class")
plt.scatter(x_neg[:,0], x_neg[:,1], color='green', label="-1 Class")

# Run the perceptron with 10 different values
boundary_label = "Perceptron Boundary"
for i in range(10):
    w_init = np.random.randn(2)
    b_init = np.random.randn()
    # Run the perceptron
    w, b = perceptron_gradient_descent(x, y, w_init, b_init)
    plot_boundary(w, b, (-5, 5), 'k-', label=boundary_label)
    boundary_label = None

# Part 3: SVM Comparison
from sklearn.svm import SVC

# Fit an SVM model
clf = SVC(kernel="linear", C=1000)
clf.fit(x, y)
w_svm = clf.coef_[0]
b_svm = clf.intercept_[0]

plot_boundary(w_svm, b_svm, (-5, 5), "r-", label="Linear SVM Boundary")

plt.title("Perceptron vs. SVM Comparison")
plt.legend()
plt.show()

# Part 4: (See writeup)

# Part 5: Adding noise
y_noisy = y.copy()

# Flip 5 points from each class
pos_indices = np.where(y == 1)[0]
flip_pos = np.random.choice(pos_indices, 5, replace=False)
y_noisy[flip_pos] = -1

neg_indices = np.where(y == -1)[0]
flip_neg = np.random.choice(neg_indices, 5, replace=False)
y_noisy[flip_neg] = 1

x_pos_noisy = x[y_noisy == 1]
x_neg_noisy = x[y_noisy == -1]

# Part 6: Soft margin SVM
# Fit and plot for each C value
c_values = [0.01, 0.1, 1, 10, 100]
for c in c_values:
    clf = SVC(kernel="linear", C=c)
    clf.fit(x, y_noisy)
    w_svm = clf.coef_[0]
    b_svm = clf.intercept_[0]
    
    plt.figure()
    
    plt.scatter(x_pos_noisy[:,0], x_pos_noisy[:,1], color='blue', label='+1')
    plt.scatter(x_neg_noisy[:,0], x_neg_noisy[:,1], color='green', label='-1')

    plot_boundary(w_svm, b_svm, (-5, 5), 'r-')
    plt.title(f"SVM with C = {c}")
    plt.show()
