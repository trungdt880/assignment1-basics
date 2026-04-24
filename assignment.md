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
