import numpy as np


def woodbury_solve(sigma_diag, X, B):
    """
    返回 (Sigma + X X^T)^{-1} B，其中 Sigma = diag(sigma_diag)。

    用 Woodbury:
        (Sigma + X X^T)^{-1} B
        = Sigma^{-1} B
          - Sigma^{-1} X (I_D + X^T Sigma^{-1} X)^{-1} X^T Sigma^{-1} B

    全程不构造 N x N 矩阵；只解一个 D x D 的系统。
    """
    sigma_diag = np.asarray(sigma_diag, dtype=float)
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)

    B_was_1d = (B.ndim == 1)
    if B_was_1d:
        B = B[:, None]                       # (N,) -> (N, 1)

    N, D = X.shape
    K = B.shape[1]

    sinv = 1.0 / sigma_diag                  # (N,)  逐行缩放系数
    SinX = sinv[:, None] * X                 # Sigma^{-1} X       (N, D)
    SinB = sinv[:, None] * B                 # Sigma^{-1} B       (N, K)

    M   = np.eye(D) + X.T @ SinX             # I_D + X^T Sigma^{-1} X  (D, D)
    rhs = X.T @ SinB                         # X^T Sigma^{-1} B        (D, K)

    Y = np.linalg.solve(M, rhs)              # (D, K)  —— 解 D x D 系统，不求逆
    out = SinB - SinX @ Y                    # (N, K)

    return out[:, 0] if B_was_1d else out


def sherman_morrison(A_inv, u, v):
    """
    返回 (A + u v^T)^{-1}，只用 A^{-1}，O(n^2)。

        (A + u v^T)^{-1}
        = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)

    分母为 0 时更新奇异：抛 ValueError。
    """
    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)

    Ainv_u  = A_inv @ u                       # (n,)
    vT_Ainv = v @ A_inv                       # (n,)
    denom   = 1.0 + float(v @ Ainv_u)         # 标量

    if denom == 0.0:
        raise ValueError("Sherman-Morrison update is singular")

    return A_inv - np.outer(Ainv_u, vT_Ainv) / denom


def logdet_rank_one_update(logdet_A, A_inv, u, v):
    """
    返回 log|A + u v^T|，已知 log|A| 与 A^{-1}。

        |A + u v^T| = (1 + v^T A^{-1} u) |A|
        取对数:  log|A + u v^T| = log|A| + log(1 + v^T A^{-1} u)

    若 1 + v^T A^{-1} u <= 0，更新后行列式不为正：抛 ValueError。
    """
    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)

    denom = 1.0 + float(v @ (A_inv @ u))

    if denom <= 0.0:
        raise ValueError("Updated determinant is not positive")

    return float(logdet_A + np.log(denom))