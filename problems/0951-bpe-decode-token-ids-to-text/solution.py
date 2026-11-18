def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    # pass
    part = []
    for i in ids:
        token = vocab[i]
        if token.startswith('G'):
            part.append(' '+token[1:])
        else:
            part.append(token)
    return "".join(part)