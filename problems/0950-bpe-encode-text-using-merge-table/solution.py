def bpe_encode(text, token_to_id, merges):
    # 1) 字符转 id
    ids = [token_to_id[ch] for ch in text]
    if not ids:
        return []

    # 2) 反复合并
    while True:
        # 找当前序列中优先级最高的可合并对（merges 里 new_id 最小）
        best_pair = None
        best_new_id = float('inf')
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            if pair in merges:
                new_id = merges[pair]
                if new_id < best_new_id:
                    best_new_id = new_id
                    best_pair = pair

        # 没有可合并的对，结束
        if best_pair is None:
            break

        a, b = best_pair
        new_id = merges[best_pair]

        # 3) 从左到右，一次性替换所有不重叠的 (a, b)
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2          # 跳过两个，不重叠
            else:
                out.append(ids[i])
                i += 1
        ids = out

    return ids