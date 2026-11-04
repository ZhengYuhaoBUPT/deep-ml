def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # Your code here
    a, b = matrix[0]
    c, d = matrix[1]
    
    # 计算行列式
    det = a * d - b * c
    
    # 判断是否奇异（由于浮点数精度，通常用很小的数作为界限，或者直接 == 0 看题目要求）
    if det == 0:
        return None
        
    # 直接返回新构造的逆矩阵，注意 b 和 c 的位置及符号
    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]