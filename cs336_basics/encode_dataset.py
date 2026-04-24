import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

tiny_data_trainpath = Path("./data/TinyStoriesV2-GPT4-train.txt")
tiny_data_valpath = Path("./data/TinyStoriesV2-GPT4-valid.txt")
owt_data_trainpath = Path("./data/owt_train.txt")
owt_data_valpath = Path("./data/owt_valid.txt")

tiny_tokenizer = Tokenizer.from_files(
    "./cs336_basics/TinyStoriesV2-GPT4-train_vocab.pkl",
    "./cs336_basics/TinyStoriesV2-GPT4-train_merges.pkl",
)

with open(tiny_data_trainpath) as f:
    tiny_samples = []
    tiny_ids = []
    with open(tiny_data_trainpath) as f:
        for _id in tiny_tokenizer.encode_iterable(f):
            tiny_ids.append(_id)

tiny_ids = np.array(tiny_ids, dtype=np.uint16)
np.save("./tiny_train_ids.npy", tiny_ids)
print("Done tiny train.")

with open(tiny_data_valpath) as f:
    tiny_samples = []
    tiny_ids = []
    with open(tiny_data_trainpath) as f:
        for _id in tiny_tokenizer.encode_iterable(f):
            tiny_ids.append(_id)

tiny_ids = np.array(tiny_ids, dtype=np.uint16)
np.save("./tiny_val_ids.npy", tiny_ids)
print("Done tiny val.")
del tiny_ids, tiny_tokenizer

owt_tokenizer = Tokenizer.from_files(
    "./cs336_basics/owt_train_vocab.pkl", "./cs336_basics/owt_train_merges.pkl"
)
with open(owt_data_trainpath) as f:
    owt_samples = []
    owt_ids = []
    with open(owt_data_trainpath) as f:
        for _id in owt_tokenizer.encode_iterable(f):
            owt_ids.append(_id)

owt_ids = np.array(owt_ids, dtype=np.uint16)
np.save("./owt_train_ids.npy", owt_ids)
print("Done owt train.")

with open(owt_data_valpath) as f:
    owt_samples = []
    owt_ids = []
    with open(owt_data_trainpath) as f:
        for _id in owt_tokenizer.encode_iterable(f):
            owt_ids.append(_id)

owt_ids = np.array(owt_ids, dtype=np.uint16)
np.save("./owt_val_ids.npy", owt_ids)
print("Done owt val.")
del owt_ids, owt_tokenizer
