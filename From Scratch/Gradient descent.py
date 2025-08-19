import numpy as np

np.random.seed(1404)
n_samples = 1000
d = 5
learning_rate = 0.0001
iterations = 300

true_weights = np.array([3.0, -2.5, 1.7, 4.2, -0.8, 2.1])  # [w0, w1, w2, w3, w4, w5]
print(f"True weights: {true_weights}")
X_b = np.random.randn(n_samples, d)
X = np.c_[np.ones((n_samples, 1)), X_b]
y = X.dot(true_weights) + np.random.randn(n_samples) * 0.5

###########################################


w = np.zeros(d + 1)  # [w0, w1, w2, ..., w5]
for i in range(iterations):
    gradient = np.zeros(d + 1)
    for j in range(n_samples):
        x_j = X[j]
        prediction = np.dot(w, x_j)
        error = y[j] - prediction
        gradient += error * x_j
    w = w + learning_rate * gradient

###########################################
print(f"T w0: {true_weights[0]:.6f} \t\t E w0: {w[0]:.6f}")
for i in range(1, len(true_weights)):
    print(f"T w{i}: {true_weights[i]:.6f} \t\t E w{i}: {w[i]:.6f}")