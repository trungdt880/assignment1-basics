import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

tiny_data_path = Path("./data/TinyStoriesV2-GPT4-train.txt")
owt_data_path = Path("./data/owt_train.txt")


with open(tiny_data_path) as f:
    tiny_samples = []
    for line in f.readlines():
        if len(tiny_samples) > 10:
            break
        if len(line) > 5:
            tiny_samples.append(line)


with open(owt_data_path) as f:
    owt_samples = f.readlines()[:10]
    for line in f.readlines():
        if len(owt_samples) > 10:
            break
        if len(line) > 5:
            owt_samples.append(line)

tiny_tokenizer = Tokenizer.from_files(
    "./cs336_basics/TinyStoriesV2-GPT4-train_vocab.pkl",
    "./cs336_basics/TinyStoriesV2-GPT4-train_merges.pkl",
)
owt_tokenizer = Tokenizer.from_files(
    "./cs336_basics/owt_train_vocab.pkl", "./cs336_basics/owt_train_merges.pkl"
)
tiny_samples_ids = []
owt_samples_ids = []
owt_on_tiny_samples_ids = []
for sample in tiny_samples:
    tiny_samples_ids.append(tiny_tokenizer.encode(sample))
print("Tiny tokenizer on Tiny")
num_tokens = sum([len(x.encode("utf-8")) for x in tiny_samples])
num_bytes = sum([len(x) for x in tiny_samples_ids])
print("Compression rate: ", num_bytes / num_tokens)
print("OWT tokenizer on OWT")
for sample in owt_samples:
    owt_samples_ids.append(owt_tokenizer.encode(sample))
num_tokens = sum([len(x.encode("utf-8")) for x in owt_samples])
num_bytes = sum([len(x) for x in owt_samples_ids])
print("Compression rate: ", num_bytes / num_tokens)
for sample in tiny_samples:
    owt_on_tiny_samples_ids.append(owt_tokenizer.encode(sample))
print("OWT tokenizer on Tiny")
num_tokens = sum([len(x.encode("utf-8")) for x in owt_samples])
num_bytes = sum([len(x) for x in owt_samples_ids])
print("Compression rate: ", num_bytes / num_tokens)

s = []
for sample in owt_samples:
    start = time.time()
    t = owt_tokenizer.encode(sample)
    duration = time.time() - start
    s.append((len(sample.encode("utf-8")) / duration))
avg = sum(s) / len(s)
print("avg bytes/second", sum(s) / len(s))

t = 825 * 10**9 / avg / 86400
print("PILE 825GB", t)

breakpoint()
