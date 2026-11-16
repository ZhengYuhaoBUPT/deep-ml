import numpy as np


def _as_inexact(A):
    """把整数/布尔数组转成浮点数组，避免整数除法问题；complex 保持不变。"""
    A = np.asarray(A)
    if not np.issubdtype(A.dtype, np.inexact):
        A = A.astype(float)
    return A


def partitioned_inverse(M, k):
    """
    用 Schur 补计算分块矩阵的逆。

    M = [[A, B],
         [C, D]]
    A = M[:k, :k]

    S = D - C A^{-1} B

    M^{-1} =
    [[A^{-1} + A^{-1} B S^{-1} C A^{-1},  -A^{-1} B S^{-1}],
     [-S^{-1} C A^{-1},                     S^{-1}]]
    """
    M = _as_inexact(M)
    n = M.shape[0]

    # 边界情况：没有真正的分块，直接求逆
    if k <= 0 or k >= n:
        return np.linalg.inv(M)

    A = M[:k, :k]
    B = M[:k, k:]
    C = M[k:, :k]
    D = M[k:, k:]

    try:
        A_inv = np.linalg.inv(A)
        S = D - C @ A_inv @ B
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # 如果 A 或 S 奇异，但 M 本身可逆，则退回普通求逆
        return np.linalg.inv(M)

    top_left = A_inv + A_inv @ B @ S_inv @ C @ A_inv
    top_right = -A_inv @ B @ S_inv
    bottom_left = -S_inv @ C @ A_inv
    bottom_right = S_inv

    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])


def block_slogdet(M, k):
    """
    用分块 Schur 补计算 slogdet。

    返回 (sign, logabsdet)，与 np.linalg.slogdet 一致。

    det(M) = det(A) * det(S)
    S = D - C A^{-1} B
    """
    M = _as_inexact(M)
    n = M.shape[0]

    if k <= 0 or k >= n:
        return np.linalg.slogdet(M)

    A = M[:k, :k]
    B = M[:k, k:]
    C = M[k:, :k]
    D = M[k:, k:]

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        # A 奇异时不能直接用这个 Schur 补公式，退回普通 slogdet
        return np.linalg.slogdet(M)

    S = D - C @ A_inv @ B

    sign_A, logdet_A = np.linalg.slogdet(A)
    sign_S, logdet_S = np.linalg.slogdet(S)

    return sign_A * sign_S, logdet_A + logdet_S


def bordered_inverse(A_inv, b, c):
    """
    M = [[A,   b],
         [b^T, c]]
    A_inv = A^{-1},  b 是向量,  c 是标量。
    Schur 补 s = c - b^T A^{-1} b, s=0 时 M 奇异, 抛 ValueError。
    """
    A_inv = _as_inexact(A_inv)
    b = np.asarray(b, dtype=A_inv.dtype).reshape(-1)
    c = np.asarray(c, dtype=A_inv.dtype).item()

    n = A_inv.shape[0]
    Ainvb = A_inv @ b
    s = c - b @ Ainvb

    if s == 0:
        raise ValueError("Schur complement is zero; bordered matrix is singular")

    s_inv = 1.0 / s

    M_inv = np.empty((n + 1, n + 1), dtype=A_inv.dtype)
    M_inv[:n, :n] = A_inv + s_inv * np.outer(Ainvb, Ainvb)
    M_inv[:n, n]  = -s_inv * Ainvb
    M_inv[n, :n]  = -s_inv * Ainvb
    M_inv[n, n]   =  s_inv

    return M_inv