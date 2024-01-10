# Copyright © 2023 Kensho Technologies, LLC

# map all bytes to valid utf-8 characters
# in the same way that the huggingface tokenizers byte level pre-tokenizer does
class HFEncoding:

    # translated from rust code found here:
    # https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs
    @staticmethod
    def bytes_char():

        bs = []
        bs.extend(range(ord('!'), ord('~') + 1))
        bs.extend(range(0xA1, 0xAC + 1))
        bs.extend(range(0xAE, 0xFF + 1))
        cs = [b for b in bs]

        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1

        return {bytes([f]): chr(t) for f, t in zip(bs, cs)}

    def __init__(self):
        # map any byte to the corresponding character
        self.byte_map = HFEncoding.bytes_char()
        # the inverse character to byte mapping
        self.inv_byte_map = {v: k for k, v in self.byte_map.items()}

    # convert an encoded string of our mapped characters back to the original bytes
    def to_bytes(self, s: str) -> bytes:
        return b"".join([self.inv_byte_map[c] for c in s])

    # convert a byte string into an encoded string of valid characters
    def to_encoded(self, byte_str: bytes) -> str:
        return "".join([self.byte_map[bytes([c])] for c in byte_str])
