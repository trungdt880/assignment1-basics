import os
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re
from bpe_example import train_bpe
from tqdm import tqdm


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
special_tokens = ["<|endoftext|>"]
MAX_VOCAB = 10000
fpath = "../data/TinyStoriesV2-GPT4-valid.txt"


def chunk_count(args):
    fpath, start, end = args
    counter = defaultdict(int)
    with open(fpath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        samples = re.split("|".join(map(re.escape, special_tokens)), chunk)
        samples = [re.finditer(PAT, sample) for sample in samples]
        for sample in samples:
            for token in sample:
                counter[token.group()] += 1
    return counter


num_processes = 4
with open(fpath, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
args = zip([fpath] * (len(boundaries) - 1), boundaries[:-1], boundaries[1:])
with Pool(4) as p:
    results = list(tqdm(p.imap_unordered(chunk_count, args), total=len(boundaries) - 1))

counter = defaultdict(int)
for local_counter in results:
    for k, v in local_counter.items():
        counter[k] += v


vocab, merges, new_tokens = train_bpe(
    counter,
    num_merges=MAX_VOCAB - (len(special_tokens) + 256),
    special_tokens=special_tokens,
)
print(merges)
breakpoint()
