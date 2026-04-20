from collections import Counter, defaultdict

from tqdm import trange

num_merges = 6


def count_adjacent_pair(
    counter: dict[tuple[bytes, bytes], int], token: tuple[bytes, ...], multiplier: int
):
    for pair in zip(token, token[1:]):
        counter[pair] += multiplier
    return counter


def count_adjacent_pair_v2(token_counter):
    pair_counter: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_token = defaultdict(set)
    for token, multiplier in token_counter.items():
        for pair in zip(token, token[1:]):
            pair_counter[pair] += multiplier
            pair_to_token[pair].add(token)
    return pair_counter, pair_to_token


def merge(best_pair: tuple[bytes, bytes], token_counter: dict[tuple[bytes, ...], int]):
    tokens = list(token_counter.keys())
    best_pair_merged = b"".join(best_pair)
    for token in tokens:
        i = 0
        new_token = []
        found = False
        while i < len(token):
            if (i + 1 < len(token)) and (
                token[i] == best_pair[0] and token[i + 1] == best_pair[1]
            ):
                new_token.append(best_pair_merged)
                i += 2
                found = True
            else:
                new_token.append(token[i])
                i += 1
        if found:
            freq = token_counter[token]
            del token_counter[token]
            token_counter[tuple(new_token)] = freq


def merge_v2(best_pair, token_counter, pair_counter, pair_to_token):
    # update pair_counter:
    best_pair_merged = b"".join(best_pair)
    best_pair_freq = pair_counter[best_pair]
    tokens = list(pair_to_token[best_pair])

    # for every token that has this pair
    for token in tokens:
        i = 0
        best_pair_ids = []

        # update that token to use the merged pair
        new_token = []
        while i < len(token):
            if (i + 1 < len(token)) and (
                token[i] == best_pair[0] and token[i + 1] == best_pair[1]
            ):
                new_token.append(best_pair_merged)
                best_pair_ids.append(i)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_token = tuple(new_token)

        token_freq = token_counter[token]
        del token_counter[token]
        if new_token not in token_counter:
            token_counter[new_token] = token_freq
        else:
            token_counter[new_token] += token_freq

        for pair in zip(token, token[1:]):
            if token in pair_to_token[pair]:
                pair_to_token[pair].remove(token)
        for pair in zip(new_token, new_token[1:]):
            pair_to_token[pair].add(new_token)

        for k, best_pair_idx_for_old_token in enumerate(best_pair_ids):
            best_pair_idx_for_new_token = best_pair_idx_for_old_token - k
            if best_pair_idx_for_new_token > 0:
                if k == 0 or best_pair_ids[k] != best_pair_ids[k - 1] + 2:
                    new_pair = (
                        new_token[best_pair_idx_for_new_token - 1],
                        new_token[best_pair_idx_for_new_token],
                    )
                    pair_counter[new_pair] += token_freq

                    old_pair = (
                        token[best_pair_idx_for_old_token - 1],
                        token[best_pair_idx_for_old_token],
                    )
                    pair_counter[old_pair] -= token_freq

            if best_pair_idx_for_new_token < len(new_token) - 1:
                new_pair = (
                    new_token[best_pair_idx_for_new_token],
                    new_token[best_pair_idx_for_new_token + 1],
                )
                pair_counter[new_pair] += token_freq

            if best_pair_idx_for_old_token < len(token) - 2:
                old_pair = (
                    token[best_pair_idx_for_old_token + 1],
                    token[best_pair_idx_for_old_token + 2],
                )
                pair_counter[old_pair] -= token_freq

    # update the pair counter:
    pair_counter[best_pair] -= best_pair_freq
    del pair_to_token[best_pair]


def train_bpe(input, num_merges, special_tokens):
    token_counter = {
        tuple(bytes([c]) for c in k.encode("utf-8")): v for k, v in input.items()
    }

    vocab: dict[int, bytes] = {
        idx: special_token.encode("utf-8")
        for idx, special_token in enumerate(special_tokens)
    }
    vocab.update({x + len(special_tokens): bytes([x]) for x in range(256)})

    pair_counter, pair_to_token = count_adjacent_pair_v2(
        token_counter,
    )
    merges = []
    for i in trange(num_merges, desc="BPE"):
        best_pair = max(pair_counter, key=lambda x: (pair_counter[x], x))
        new_idx = len(vocab)
        vocab[new_idx] = b"".join(best_pair)
        merges.append(best_pair)
        # merge(best_pair, token_counter, pair_counter)
        merge_v2(best_pair, token_counter, pair_counter, pair_to_token)
    return vocab, merges, token_counter


if __name__ == "__main__":
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    input = Counter(text.split(" "))
    vocab, merges, tokens = train_bpe(
        input, num_merges=6, special_tokens=["<|endoftext|>"]
    )
    print(merges)
