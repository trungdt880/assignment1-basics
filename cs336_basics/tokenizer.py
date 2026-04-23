import pickle as pkl
from collections.abc import Iterable
from pathlib import Path

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.special_tokens = []
        if special_tokens is not None:
            special_tokens = sorted(set(special_tokens), key=len, reverse=True)
            for special_token in special_tokens:
                special_token_encoded = special_token.encode("utf-8")
                if special_token_encoded not in self.vocab_inv:
                    l = len(self.vocab)
                    self.vocab[l] = special_token_encoded
                    self.vocab_inv[special_token_encoded] = l
                self.special_tokens.append(special_token)
        self.special_tokens_set = set(self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab_filepath: Path = Path(vocab_filepath)
        merges_filepath: Path = Path(merges_filepath)
        assert vocab_filepath.exists() and merges_filepath.exists()

        with open(vocab_filepath, "rb") as f:
            vocab = pkl.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pkl.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self.special_tokens:
            pattern = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            samples = re.split(pattern, text)
        else:
            samples = [text]

        result = []
        for sample in samples:
            if sample in self.special_tokens_set:
                result.append(self.vocab_inv[sample.encode("utf-8")])
            else:
                sample = re.finditer(self.pattern, sample)
                for word in sample:
                    word = [bytes([x]) for x in word.group().encode("utf-8")]

                    while len(word) > 1:
                        best_idx = -1
                        best_rank = float("inf")
                        best_pair = None

                        for pair_idx, pair in enumerate(zip(word, word[1:])):
                            pair_rank = self.merge_rank.get(pair, -1)
                            if pair_rank == -1:
                                continue
                            if pair_rank < best_rank:
                                best_idx = pair_idx
                                best_rank = pair_rank
                                best_pair = pair

                        # can't merge anymore
                        if best_idx == -1:
                            break

                        # if found -> merge
                        word[best_idx] = b"".join(best_pair)
                        word.pop(best_idx + 1)
                    for subword in word:
                        id = self.vocab_inv.get(subword, None)
                        if id is None:
                            raise ValueError(f"Subword not in vocab: {subword!r}")
                        result.append(id)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        return iter([0])

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        result = []
        for id in ids:
            if id not in self.vocab:
                raise Exception("Invalid token id")
            result.append(self.vocab[id])
        result = b"".join(result).decode("utf-8", errors="replace")

        return result


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "./cs336_basics/TinyStoriesV2-GPT4-train_vocab.pkl",
        "./cs336_basics/TinyStoriesV2-GPT4-train_merges.pkl",
        # ["<|endoftext|>"],
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    s = "the cat ate<|endoftext|>the dog in the sae wants to kill you<|endoftext|>"
    s = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    s = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # s = ""
    encoded_ids = tokenizer.encode(s)
    print(f"Encode: {encoded_ids}")
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    print(f"{tokenized_string=}")
    decoded = tokenizer.decode(encoded_ids)
    print(f"Decode: {decoded}")
