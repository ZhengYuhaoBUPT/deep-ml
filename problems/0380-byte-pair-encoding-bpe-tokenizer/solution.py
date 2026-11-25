def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """

    words = []
    for text, freq in corpus.items():
        token = text.split()
        if token:
            words.append([token, freq])

    merges = []

    for _ in range(num_merges):
        counts = {}
        for tokens, freq in words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                counts[pair] = counts.get(pair,0) + freq
        
        if not counts:
            break
        
        best_pair = None
        best_count = -1
        for pair,cnt in counts.items():
            if cnt > best_count:
                best_count = cnt
                best_pair = pair

        merges.append(best_pair)

        a,b = best_pair
        merged = a+b

        new_words = []
        for tokens, freq in words:
            out = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            new_words.append([out,freq])

        words = new_words
    
    return merges




