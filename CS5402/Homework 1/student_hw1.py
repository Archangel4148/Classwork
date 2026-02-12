import numpy as np
import matplotlib.pyplot as plt
# DO NOT import any other libraries.


def confusion_matrix_metrics(conf_mat):
    """
    Compute evaluation metrics from a confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray of shape (K, K)
        Confusion matrix where conf_mat[i, j] is the number of
        samples with true label i predicted as label j.

    Returns
    -------
    metrics : np.ndarray of shape (K, 4)
        Each row corresponds to one class.
        Columns are:
        [accuracy, recall, precision, f1_score]
    """

    k = conf_mat.shape[0]
    # i, i = TP
    # i, ~i = FN (sum)
    # ~i, i = FP (sum)
    # ~i, ~i = TN (sum)

    # Build metrics for each class
    metrics = []
    for cls in range(k):
        row = conf_mat[cls]
        col = conf_mat[:, cls]

        # Get TP, FN, FP, and TN counts
        tp = row[cls]
        fn = np.sum(row) - tp
        fp = np.sum(col) - tp
        tn = np.sum(conf_mat) - tp - fn - fp  # Everything else is TN

        # Calculate metrics
        try:
            accuracy = (tn + tp) / (tp + fn + fp + tn)
        except ZeroDivisionError:
            accuracy = 0.0
        try:
            recall = tp / (fn + tp)
        except ZeroDivisionError:
            recall = 0.0
        try:
            precision = tp / (fp + tp)
        except ZeroDivisionError:
            precision = 0.0
        try:
            f1_score = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0.0

        metrics.append([accuracy, recall, precision, f1_score])
    return np.array(metrics)




def plot_roc_curve(y_true, y_score, num_thresholds=100):
    """
    Plot ROC curve from ground-truth labels and predicted probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1)

    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the input

    num_thresholds : int
        Number of thresholds to evaluate (default=100)

    Returns
    -------
    fpr : np.ndarray
        False Positive Rates

    tpr : np.ndarray
        True Positive Rates
    """


    tpr_list = []
    fpr_list = []

    # TODO:
    # Your implementation.
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    for threshold in thresholds:
        # Evaluate TPR and FPR with threshold, and append to the lists
        pred = y_score >= threshold
        # Get counts
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))

        # Calculate TP Rate and FP Rate
        try:
            tpr = tp / (fn + tp)
        except ZeroDivisionError:
            tpr = 0.0

        try:
            fpr = fp / (fp + tn)
        except ZeroDivisionError:
            fpr = 0.0

        # Add the rates to the lists
        tpr_list.append(tpr)
        fpr_list.append(fpr)


    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_list, tpr_list, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return np.array(fpr_list), np.array(tpr_list)
