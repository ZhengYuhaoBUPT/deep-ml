import numpy as np

def lu_decomposition(A: list) -> tuple:
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        if abs(U[i, i]) < 1e-12:
            raise ValueError("Zero pivot; matrix needs pivoting or is singular")
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]

    return L, U