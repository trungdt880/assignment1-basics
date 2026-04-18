# unicode1
a. \x00
b. repr shows the raw value, so we see the \x00 which is the official string, whereas print (str) would be more user-friendly, in this case non-printing null char
c. as above, when use print it would use str(), hence it would be empty (non-printing null char)

# unicode2
a. UTF-8 code unit size only 8 bit [0,255], also char can have length varied (1-4), so memory would take less comparing to UTF-16 (16 bit; bytes per char 2-4) or UTF-32 (32-bit; fixed bytes per char 4)
b. wrong because one char can have varied length of byte.
c. b'\xC0\xAF'; which represent '/'. Invalid because UTF-8 prioritize as less byte as possible. In this case, '/' can be presented with only 1 byte (b'\x2F').
NOTE: UTF-4 has bytes-per-char 1-4 -> will be four pattern for 1/2/3/4-byte sequence. each byte will have some prefix bits for faster parse; other will be used for decode.
