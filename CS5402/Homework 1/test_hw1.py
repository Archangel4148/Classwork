import numpy as np
# -----------------------------
# Import student submission
# -----------------------------
import student_hw1 as hw


# =====================================================
# Task 1: Confusion Matrix Metrics
# =====================================================
# Case-1
conf = np.array([[10, 0],
                 [0, 15]])

out = hw.confusion_matrix_metrics(conf)
print(out)


# Case-2:
conf = np.array([[40, 5, 5],
                 [6, 30, 4],
                 [3, 7, 20]])

out = hw.confusion_matrix_metrics(conf)
print(out)

# Case-3:
np.random.seed(0)
conf = np.random.randint(0, 100, size=(5, 5))

out = hw.confusion_matrix_metrics(conf)
print(out)


# =====================================================
# Task 2: ROC Curve
# =====================================================
# Case-1: Simple binary classification
y_true  = np.array([0, 0, 0, 1, 1, 1])
y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])

fpr, tpr = hw.plot_roc_curve(y_true, y_score)


# Case-2: Random binary classification
np.random.seed(1)
y_true = np.random.randint(0, 2, 100)
y_score = np.random.rand(100)

fpr, tpr = hw.plot_roc_curve(y_true, y_score)

