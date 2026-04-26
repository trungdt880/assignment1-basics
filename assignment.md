# unicode1
a. \x00
b. repr shows the raw value, so we see the \x00 which is the official string, whereas print (str) would be more user-friendly, in this case non-printing null char
c. as above, when use print it would use str(), hence it would be empty (non-printing null char)

# unicode2
a. UTF-8 code unit size only 8 bit [0,255], also char can have length varied (1-4), so memory would take less comparing to UTF-16 (16 bit; bytes per char 2-4) or UTF-32 (32-bit; fixed bytes per char 4)
b. wrong because one char can have varied length of byte.
c. b'\xC0\xAF'; which represent '/'. Invalid because UTF-8 prioritize as less byte as possible. In this case, '/' can be presented with only 1 byte (b'\x2F').
NOTE: UTF-4 has bytes-per-char 1-4 -> will be four pattern for 1/2/3/4-byte sequence. each byte will have some prefix bits for faster parse; other will be used for decode.

# train_bpe_tinystories
a. 1min44. Longest token: " accomplishment". yes it makes sense
b. getting the index of best pair by max() takes the longest

# train_bpe_expts_owt
a. 3h33m. longest token: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' -> 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'
b. Much larger size, much more unicode tokens

# tokenizer_experiments
a/b. 
Tiny tokenizer on Tiny
Compression rate:  0.24891101431238333
OWT tokenizer on OWT
Compression rate:  0.22816901408450704
OWT tokenizer on Tiny
Compression rate:  0.22816901408450704
avg bytes/second 634835.0885151816
c. PILE 825GB would take 15.04 days
d. because tokenized ID is unsigned int, and max vocab is 32K which is way smaller than uint16 max.

# transformer_accounting
## For one transformer block:
- QKVO projections: 4*(2*((model_dim)**2*seq_len))
- Attention: 2*(2*seq_len*seq_len*d_model)
- FFN: 3*(2*d_model*seq_len*d_ff)
- FLOPS_ONE_BLOCK=4*(2*((model_dim)**2*seq_len)) + 2*(2*seq_len*seq_len*d_model) + 3*(2*d_model*seq_len*d_ff)
## Transformer
- FLOPS_HEAD=2*seq_len*d_model*vocab_size
- FLOPS_LAYER=num_layers * FLOPS_ONE_BLOCK
- TOTAL=FLOPS_LAYER+FLOPS_HEAD

```text
================================================================================
Config
d_model=1600
d_ff=4288
vocab_size=50257
seq_len=1024
num_layers=48
================================================================================
(a) PARAMS
TOKEN_EMBED=321,644,800
QKVO=40,960,000
FFN=82,329,600
RMS_NORM_IN_ONE_BLOCK=12,800
ONE_BLOCK=123,302,400
RMS_NORM_FINAL=6,400
TOTAL_PARAMS=1,640,452,800.0
TOTAL_PARAMS_BYTES=6,561,811,200
================================================================================
(b) FLOPS
FLOPS_QKVO=20,971,520,000
FLOPS_ATTN=6,710,886,400
FLOPS_FFN=42,152,755,200
FLOPS_HEAD=164,682,137,600
FLOPS_LAYER=3,352,087,756,800
TOTAL_FLOPS=3,516,769,894,400
(c) For a block, FFN takes the largest
(d) Ablate config: FFN + HEAD takes more
================================================================================
Config small
d_ff=2048
d_model=768
num_layers=12
========================================
FLOPS
FLOPS_QKVO=4,831,838,208, 0.02
FLOPS_ATTN=3,221,225,472, 0.01
FLOPS_FFN=9,663,676,416, 0.03
FLOPS_ONE_BLOCK=17,716,740,096, 0.06
FLOPS_LAYERS=212,600,881,152, 0.73
FLOPS_HEAD=79,047,426,048, 0.27
TOTAL_FLOPS=291,648,307,200
================================================================================
Config medium
d_ff=2752
d_model=1024
num_layers=24
========================================
FLOPS
FLOPS_QKVO=8,589,934,592, 0.01
FLOPS_ATTN=4,294,967,296, 0.01
FLOPS_FFN=17,314,086,912, 0.02
FLOPS_ONE_BLOCK=30,198,988,800, 0.04
FLOPS_LAYERS=724,775,731,200, 0.87
FLOPS_HEAD=105,396,568,064, 0.13
TOTAL_FLOPS=830,172,299,264
================================================================================
Config large
d_ff=3456
d_model=1280
num_layers=36
========================================
FLOPS
FLOPS_QKVO=13,421,772,800, 0.01
FLOPS_ATTN=5,368,709,120, 0.00
FLOPS_FFN=27,179,089,920, 0.02
FLOPS_ONE_BLOCK=45,969,571,840, 0.03
FLOPS_LAYERS=1,654,904,586,240, 0.93
FLOPS_HEAD=131,745,710,080, 0.07
TOTAL_FLOPS=1,786,650,296,320
(e)
3,516,769,894,400 -> 133,577,729,638,400
BEFORE:
FLOPS_QKVO=20,971,520,000, 0.01
FLOPS_ATTN=6,710,886,400, 0.00
FLOPS_FFN=42,152,755,200, 0.01
FLOPS_ONE_BLOCK=69,835,161,600, 0.02
FLOPS_LAYERS=3,352,087,756,800, 0.95
FLOPS_HEAD=164,682,137,600, 0.05
TOTAL_FLOPS=3,516,769,894,400
AFTER:
FLOPS_QKVO=335,544,320,000, 0.00
FLOPS_ATTN=1,717,986,918,400, 0.01
FLOPS_FFN=674,444,083,200, 0.01
FLOPS_ONE_BLOCK=2,727,975,321,600, 0.02
FLOPS_LAYERS=130,942,815,436,800, 0.98
FLOPS_HEAD=2,634,914,201,600, 0.02
TOTAL_FLOPS=133,577,729,638,400

```

# learning_rate_tuning
- converge speed 1e2 > 1e1
- 1e3 diverge
