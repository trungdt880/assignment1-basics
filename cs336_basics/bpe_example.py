from collections import defaultdict

text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

tokens = list(map(lambda x: tuple([y.encode("utf-8") for y in x]), text.split(" ")))
vocab: dict[int, bytes] = {x + 1: bytes([x]) for x in range(256)}  # index -> bytes
vocab[0] = "<|endoftext|>".encode("utf-8")

num_merges = 6


def count_adjacent_pair(
    counter: dict[tuple[bytes, bytes], int], token: tuple[bytes, ...]
):
    for pair in zip(token, token[1:]):
        counter[pair] += 1
    return counter


def merge(best_pair: tuple[bytes, bytes], tokens: list[tuple[bytes, ...]]):
    for token_idx, token in enumerate(tokens):
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
        tokens[token_idx] = tuple(new_token)


for i in range(num_merges):
    counter: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for token in tokens:
        count_adjacent_pair(counter, token)
    print(counter)
    best_pair = max(counter, key=lambda x: (counter.get(x), x))
    new_idx = len(vocab)
    vocab[new_idx] = b"".join(best_pair)
    merge(best_pair, tokens)
    print(best_pair)
    print(tokens)
    print("-")
print(vocab)
