# --- cell 1 ---
def train_tokenizer(corpus, vocab_size):
    from collections import Counter

    # ---------- 1. 字符级基础词表 ----------
    chars = sorted(set("".join(corpus)))
    if len(chars) > vocab_size:          # 极端防御
        chars = chars[:vocab_size]

    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_str  = {i: c for i, c in enumerate(chars)}

    # 语料转为字符 id 序列
    words = [[char_to_id[c] for c in name] for name in corpus]

    # ---------- 2. BPE 合并 ----------
    merges = []                          # [(a, b, new_id)]，按优先级（插入顺序）
    next_id = len(chars)

    while next_id < vocab_size:
        # 统计相邻对出现次数（同一词内多次出现多次计数）
        pair_counts = Counter()
        for tokens in words:
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1

        if not pair_counts:
            break

        (a, b), _ = pair_counts.most_common(1)[0]
        new_id = next_id

        # 注册新 token 字符串
        id_to_str[new_id] = id_to_str[a] + id_to_str[b]
        merges.append((a, b, new_id))
        next_id += 1

        # 全量合并（从左到右，不重叠）
        new_words = []
        for tokens in words:
            out = []
            i, L = 0, len(tokens)
            while i < L:
                if i + 1 < L and tokens[i] == a and tokens[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            new_words.append(out)
        words = new_words

    # ---------- 3. encode ----------
    def encode(text):
        tokens = [char_to_id[c] for c in text]
        # 按优先级顺序依次应用每个 merge
        for a, b, new_id in merges:
            if len(tokens) < 2:
                break
            out = []
            i, L = 0, len(tokens)
            while i < L:
                if i + 1 < L and tokens[i] == a and tokens[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    # ---------- 4. decode ----------
    def decode(ids):
        return "".join(id_to_str[i] for i in ids)

    return encode, decode