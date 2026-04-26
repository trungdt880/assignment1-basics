from cs336_basics.models.common import round_up_to_multiple

d_model = 1600
vocab_size = 50257
seq_len = 16384
seq_len = 1024
num_layers = 48

print(f"{vocab_size=}")
print(f"{seq_len=}")
for name, num_layers, d_model in [
    ("small", 12, 768),
    ("medium", 24, 1024),
    ("large", 36, 1280),
    ("XL", 48, 1600),
]:
    d_ff = round_up_to_multiple(8 / 3 * d_model, 64)

    print("=" * 80)
    print(f"Config {name}")
    print(f"{d_ff=}")
    print(f"{d_model=}")
    print(f"{num_layers=}")

    print("=" * 40)
    print("PARAMS")
    FLOAT_SIZE = 4  # 4Bytes -> 32bits
    TOKEN_EMBED = FLOAT_SIZE * vocab_size * d_model
    HEAD = FLOAT_SIZE * vocab_size * d_model
    QKVO = 4 * (FLOAT_SIZE * d_model * d_model)
    FFN = 3 * (FLOAT_SIZE * d_model * d_ff)
    RMS_NORM_IN_ONE_BLOCK = 2 * (FLOAT_SIZE * d_model)
    ONE_BLOCK = QKVO + FFN + RMS_NORM_IN_ONE_BLOCK
    RMS_NORM_FINAL = 1 * (FLOAT_SIZE * d_model)
    TOTAL_PARAMS_BYTES = TOKEN_EMBED + num_layers * ONE_BLOCK + RMS_NORM_FINAL + HEAD
    TOTAL_PARAMS = TOTAL_PARAMS_BYTES / FLOAT_SIZE
    print(f"{TOKEN_EMBED=:,}")
    print(f"{QKVO=:,}")
    print(f"{FFN=:,}")
    print(f"{RMS_NORM_IN_ONE_BLOCK=:,}")
    print(f"{ONE_BLOCK=:,}")
    print(f"{RMS_NORM_FINAL=:,}")
    print(f"{TOTAL_PARAMS=:,}")
    print(f"{TOTAL_PARAMS_BYTES=:,}")

    print("=" * 40)
    print("FLOPS")
    FLOPS_QKVO = 4 * (2 * ((d_model) ** 2 * seq_len))
    FLOPS_ATTN = 2 * (2 * seq_len * seq_len * d_model)
    FLOPS_FFN = 3 * (2 * d_model * seq_len * d_ff)
    FLOPS_ONE_BLOCK = FLOPS_QKVO + FLOPS_ATTN + FLOPS_FFN
    FLOPS_HEAD = 2 * seq_len * d_model * vocab_size
    FLOPS_LAYERS = num_layers * FLOPS_ONE_BLOCK
    TOTAL_FLOPS = FLOPS_LAYERS + FLOPS_HEAD
    print(f"{FLOPS_QKVO=:,}, {FLOPS_QKVO/TOTAL_FLOPS:.2f}")
    print(f"{FLOPS_ATTN=:,}, {FLOPS_ATTN/TOTAL_FLOPS:.2f}")
    print(f"{FLOPS_FFN=:,}, {FLOPS_FFN/TOTAL_FLOPS:.2f}")
    print(f"{FLOPS_ONE_BLOCK=:,}, {FLOPS_ONE_BLOCK/TOTAL_FLOPS:.2f}")
    print(f"{FLOPS_LAYERS=:,}, {FLOPS_LAYERS/TOTAL_FLOPS:.2f}")
    print(f"{FLOPS_HEAD=:,}, {FLOPS_HEAD/TOTAL_FLOPS:.2f}")
    print(f"{TOTAL_FLOPS=:,}")
