# linear-regression-practice

Questions:
- What's the difference between element-wise vector/matrix multiplication and dot product?

Using `.dot` performs matrix multiplication whereas the `*` performs element-wise multiplication. `*` expects each matrix to be the exact same shape. `.dot` will allow different shape matrices to by multiplied together as long as the number of columns in the first matrix matches the number of rows in the second matrix.

```
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6], [7, 8]])
>>> a * b
array([[ 5, 12],
       [21, 32]])
>>> a.dot(b)
array([[19, 22],
       [43, 50]])
```

In the Coursera exercises, you saw `h_theta(x) = theta' * x`.
This is actually referring to _1 row of the X matrix_ (that's why 'x' is lowercase).
This implementation assumes you're inside a `for` loop doing: 
```python
result = []
for i in range(m):
    # theta.transpose().dot(X[0]) # result to be [ [0.] ]
    # theta.transpose().dot(X[1]) # result to be [ [0.], [0.] ]
    # theta.transpose().dot(X[2]) # result to be [ [0.], [0.], [0.] ]
    # etc..
    theta.transpose().dot(X[i])
```

The vectorized implementation does the above multiplication in 1 operation.

```python
X.shape     # (97, 2)
theta.shape # (2, 1)

# the number of columns in 'X' matches the number of rows in 'theta', so we can perform matrix multiplication with .dot!
h = X.dot(theta)
```

- When computing the gradient, why do we not need to use np.sum?
- How do you plot the regression line after you finish modifying theta?
- How do you get all the values of a single column from a (n, 2+) shaped matrix in numpy?
