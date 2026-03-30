import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 1. Generate a complex, non-linear regression dataset
X, y = make_friedman1(n_samples=2000, noise=5.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    print(f"{name:<25} | Train MSE: {train_mse:>7.2f} | Test MSE: {test_mse:>7.2f}")

print("-" * 65)
print("Part A: The Baselines")
print("-" * 65)
# High Variance Base Learner
deep_tree = DecisionTreeRegressor(max_depth=None, random_state=42)
evaluate_model("1. Single Deep Tree", deep_tree)

# High Bias Base Learner
stump = DecisionTreeRegressor(max_depth=1, random_state=42)
evaluate_model("2. Single Decision Stump", stump)

print("\n" + "-" * 65)
print("Part B: The 'Correct' Ensembles (Theoretical Best Practice)")
print("-" * 65)
# Bagging loves High Variance base learners
bagged_deep_trees = BaggingRegressor(estimator=deep_tree, n_estimators=100, random_state=42, n_jobs=-1)
evaluate_model("3. Bagged Deep Trees", bagged_deep_trees)

# Boosting loves High Bias base learners
boosted_stumps = GradientBoostingRegressor(max_depth=1, n_estimators=100, learning_rate=0.1, random_state=42)
evaluate_model("4. Boosted Stumps", boosted_stumps)

print("\n" + "-" * 65)
print("Part C: The 'Role Reversal' (Intentional Failure)")
print("-" * 65)
# Bagging a High Bias model
bagged_stumps = BaggingRegressor(estimator=stump, n_estimators=100, random_state=42, n_jobs=-1)
evaluate_model("5. Bagged Stumps", bagged_stumps)

# Boosting a High Variance model
boosted_deep_trees = GradientBoostingRegressor(max_depth=15, n_estimators=100, learning_rate=0.1, random_state=42)
evaluate_model("6. Boosted Deep Trees", boosted_deep_trees)
print("-" * 65)