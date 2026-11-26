def make_input_target(token_ids, context_size):
    """Build next-token prediction (input, target) pair.

    Args:
        token_ids: list[int] of token IDs (length >= context_size + 1)
        context_size: int, number of tokens in the input window
    Returns:
        list: [x, y] where x and y are lists of length context_size,
              with y being x shifted right by one position.
    """
    if len(token_ids) < context_size + 1:
        raise ValueError("token_ids must have at least context_size + 1 elements")
    x = token_ids[:context_size]          # 前 context_size 个
    y = token_ids[1:context_size + 1]     # 右移一位：下标 1..context_size
    return [x, y]