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


def get_pair_from_word(word):
    return [(x, y) for x, y in zip(word, word[1:])]


def get_pair_byte_from_word(word):
    return [(x.encode("utf-8"), y.encode("utf-8")) for x, y in zip(word, word[1:])]


def merge_v2(
    best_pair: tuple[bytes, bytes],
    word_counter: dict[tuple[bytes, ...], int],
    pair_counter: dict[tuple[bytes, bytes], int],
    pair_to_word: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
):
    best_pair_merged = b"".join(best_pair)

    affected_words = list(pair_to_word[best_pair])
    for word in affected_words:
        word_freq = word_counter.get(word, 0)
        if word_freq <= 0:
            continue

        old_word_pair_count = Counter(get_pair_from_word(word))
        for pair, pair_freq in old_word_pair_count.items():
            # how many times word appear * how many time that pair appear in that word
            pair_counter[pair] -= word_freq * pair_freq
            if pair_counter[pair] <= 0:
                pair_counter.pop(pair, None)
            pair_to_word[pair].discard(word)
            if not pair_to_word[pair]:
                pair_to_word.pop(pair, None)

        new_word = []
        i = 0
        while i < len(word):
            if (
                (i < len(word) - 1)
                and word[i] == best_pair[0]
                and word[i + 1] == best_pair[1]
            ):
                new_word.append(best_pair_merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)

        word_counter.pop(word)
        word_counter[new_word] = word_counter.get(new_word, 0) + word_freq

        new_word_pair_count = Counter(get_pair_from_word(new_word))
        for pair, pair_freq in new_word_pair_count.items():
            # how many times word appear * how many time that pair appear in that word
            pair_counter[pair] += word_freq * pair_freq
            pair_to_word[pair].add(new_word)


def train_bpe_internal(input, num_merges, special_tokens):
    word_counter = {
        tuple(bytes([c]) for c in k.encode("utf-8")): v for k, v in input.items()
    }

    vocab: dict[int, bytes] = {
        idx: special_token.encode("utf-8")
        for idx, special_token in enumerate(special_tokens)
    }
    vocab.update({x + len(special_tokens): bytes([x]) for x in range(256)})

    pair_counter, pair_to_token = count_adjacent_pair_v2(
        word_counter,
    )
    merges = []
    for i in trange(num_merges, desc="BPE"):
        best_pair = max(pair_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        new_idx = len(vocab)
        vocab[new_idx] = b"".join(best_pair)
        merges.append(best_pair)
        # merge(best_pair, token_counter, pair_counter)
        merge_v2(best_pair, word_counter, pair_counter, pair_to_token)
    return vocab, merges, word_counter


if __name__ == "__main__":
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    input = Counter(text.split(" "))
    vocab, merges, tokens = train_bpe_internal(
        input, num_merges=6, special_tokens=["<|endoftext|>"]
    )
    print(merges)
