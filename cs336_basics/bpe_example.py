from collections import Counter, defaultdict

num_merges = 6


def count_adjacent_pair(
    counter: dict[tuple[bytes, bytes], int], token: tuple[bytes, ...], multiplier: int
):
    for pair in zip(token, token[1:]):
        counter[pair] += multiplier
    return counter


def merge(best_pair: tuple[bytes, bytes], token_counter: dict[tuple[bytes, ...], int]):
    tokens = list(token_counter.keys())
    for token in tokens:
        i = 0
        new_token = []
        while i < len(token):
            if (i + 1 < len(token)) and (
                token[i] == best_pair[0] and token[i + 1] == best_pair[1]
            ):
                new_token.append(b"".join(best_pair))
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        freq = token_counter[token]
        del token_counter[token]
        token_counter[tuple(new_token)] = freq


def train_bpe(input, num_merges):
    token_counter = {
        tuple(bytes([c]) for c in k.encode("utf-8")): v for k, v in input.items()
    }

    vocab: dict[int, bytes] = {x + 1: bytes([x]) for x in range(256)}  # index -> bytes
    vocab[0] = "<|endoftext|>".encode("utf-8")
    merges = {}
    for i in range(num_merges):
        counter: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for token, multiplier in token_counter.items():
            count_adjacent_pair(counter, token, multiplier)
        best_pair = max(counter, key=lambda x: (counter.get(x), x))
        new_idx = len(vocab)
        vocab[new_idx] = b"".join(best_pair)
        merges[best_pair] = vocab[new_idx]
        merge(best_pair, token_counter)
    return vocab, merges, token_counter


if __name__ == "__main__":
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    input = Counter(text.split(" "))
    vocab, merges, tokens = train_bpe(input, num_merges=6)
    print(merges)
