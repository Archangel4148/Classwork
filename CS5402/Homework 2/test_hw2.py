import math
import numpy as np
import matplotlib.pyplot as plt
import random
from student_hw2 import information_gain, perceptron_gradient_descent


## Task-1:
dataset1 = [
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Mild', 'No'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rain', 'Cool', 'Yes']
]

print(information_gain(dataset1))

dataset2=[
        ['A', 'X', 'Yes'],
        ['A', 'Y', 'Yes'],
        ['B', 'X', 'No'],
        ['B', 'Y', 'No']
    ]

print(information_gain(dataset2))




## Task-2:
X = [
    [2, 2],   # ID 0
    [1, -1],  # ID 1
    [-2, -1]  # ID 2
]

y = [1, -1, -1]
w_init = [0, 0]
b_init = 0
lr = 1.0
max_iter = 10

w_final, b_final = perceptron_gradient_descent(X, y, w_init, b_init, lr, max_iter)
print("Final weights:", w_final)
print("Final bias:", b_final)


# Non-separable case
X_nonsep = [
    [1, 1],    # ID 0
    [1, -1],   # ID 1
    [-1, 1],   # ID 2
    [-1, -1]   # ID 3
]

y_nonsep = [1, -1, -1, 1]
# Perceptron will not converge, but should run for max_iter iterations
w_final_nonsep, b_final_nonsep = perceptron_gradient_descent(X_nonsep, y_nonsep, w_init, b_init, lr, max_iter)
print("Final weights (non-separable):", w_final_nonsep)
print("Final bias (non-separable):", b_final_nonsep)