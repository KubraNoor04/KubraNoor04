import numpy as np

def gradient_descent(x, y):
    m = 0
    c = 0
    learning_rate = 0.01
    iterations = 100
    n = len(x)

    for i in range(iterations):
        y_predicted = m * x + c
        cost = (1 / (2 * n)) * sum([val**2 for val in (y - y_predicted)])  # Mean squared error to find cost
        md = -(1 / n) * sum(x * (y - y_predicted))  # Partial derivative with respect to m
        cd = -(1 / n) * sum(y - y_predicted)  # Partial derivative with respect to c
        m = m - learning_rate * md  # Update m
        c = c - learning_rate * cd  # Update c
        print("Iteration:", i + 1)
        print("m:", m)
        print("c:", c)
        print("Cost:", cost)
        print("\n")

x = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90,100])

gradient_descent(x, y)
