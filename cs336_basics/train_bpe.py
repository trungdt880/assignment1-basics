import pickle as pkl
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import tyro
from tqdm import tqdm

from cs336_basics.bpe_example import _train_bpe
from cs336_basics.pretokenization_example import chunk_count, find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: Path = Path("./data/TinyStoriesV2-GPT4-train.txt"),
    vocab_size: int = 10000,
    special_tokens: list[str] = ["<|endoftext|>"],
    output_path: Path | None = None,
):
    num_processes = 16
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    args = zip(
        [(input_path, special_tokens, PAT)] * (len(boundaries) - 1),
        boundaries[:-1],
        boundaries[1:],
    )
    with Pool(num_processes) as p:
        results = list(
            tqdm(p.imap_unordered(chunk_count, args), total=len(boundaries) - 1)
        )

    counter = defaultdict(int)
    for local_counter in results:
        for k, v in local_counter.items():
            counter[k] += v

    vocab, merges, new_tokens = _train_bpe(
        counter,
        num_merges=vocab_size - (len(special_tokens) + 256),
        special_tokens=special_tokens,
    )
    if output_path:
        with open(output_path / f"{input_path.stem}_vocab.pkl", "wb") as f:
            pkl.dump(vocab, f)
        with open(output_path / f"{input_path.stem}_merges.pkl", "wb") as f:
            pkl.dump(merges, f)
        with open(output_path / f"{input_path.stem}_tokens.pkl", "wb") as f:
            pkl.dump(new_tokens, f)
    return vocab, merges


if __name__ == "__main__":
    tyro.cli(train_bpe)
