# Copyright © 2023 Kensho Technologies, LLC

from typing import List, Dict, Union, Tuple

import numpy as np

from .HFEncoding import HFEncoding


Tokenizable = Union[str,bytes]


class SaGeTokenizer:

    def __init__(self, initial_vocabulary, max_len: int=16):
        self.hfe = HFEncoding()
        self.byte_vocab: Dict[bytes, int]     = None
        self.inv_byte_vocab: Dict[int, bytes] = None
        self.str_vocab: Dict[str, int]        = None
        self.inv_str_vocab: Dict[int, str]    = None

        self.set_vocabulary(initial_vocabulary)
        self.max_len = max_len

    # given an order list of bytes for vocabulary
    # create our internal structures
    # overwriting any previous values
    def set_vocabulary(self, new_vocab: List[bytes]):
        self.byte_vocab = self.set_bytes_vocab(new_vocab)

        # make sure we always have all single bytes in vocabulary
        verify_all_single_byte_exist_in_vocab(self.byte_vocab)

        # inverted map of inv_byte_vocab (int index : bytes)
        self.inv_byte_vocab = {v: k for (k, v) in self.byte_vocab.items()}

        # encoded str : int index
        # convert bytes to our encoded form for keys
        self.str_vocab = {self.hfe.to_encoded(k): v for (k, v) in self.byte_vocab.items()}
        # int index : encoded str
        self.inv_str_vocab = {v: k for (k, v) in self.str_vocab.items()}

    @staticmethod
    def set_bytes_vocab(new_vocab: List[bytes]) -> Dict[bytes, int]:
        # bytes : int index
        byte_vocab = {}
        for idx, token in enumerate(new_vocab):
            # token should have been converted to bytes
            assert type(token) == bytes
            byte_vocab[token] = idx
        return byte_vocab

    def id_to_bytes(self, token_id: int) -> bytes:
        return self.inv_byte_vocab[token_id]

    def id_to_encoded(self, token_id: int) -> str:
        return self.inv_str_vocab[token_id]

    def get_vocabulary(self) -> Dict[bytes, int]:
        return self.byte_vocab

    def vocab_size(self) -> int:
        return len(self.byte_vocab)

    def print_tokens(self, ids: List[int]) -> List[bytes]:
        """
        Human readable for debugging
        """
        return [self.inv_byte_vocab[i] for i in ids]

    def add_all_byte_ids(self, vocab: Dict[int,float], score: float=1e400):
        """
        For each single byte, look up its id, then assign the given score to that id in the given dictionary.
        """
        for i in range(256):
            # what is the corresponding token id
            tid = self.byte_vocab[bytes([i])]
            # add that with a "good" score
            vocab[tid] = score

    def tokenize(self, sent: Tokenizable, tokens_only: bool=False) -> Union[List[int], List[Tuple[int, int, int]]]:
        """
        Split the gives sentence into tokens and convert them to IDs.
        """
        if isinstance(sent, str):
            sent = bytes(sent, encoding='utf-8')
        data = []
        i = 0
        while i < len(sent):  # Iterate through the sentence input
            for j in range(self.max_len, 0, -1):  # Find the longest possible token
                tok = sent[i:i + j]
                if tok in self.byte_vocab:
                    if tokens_only:
                        # Only add token_id to results
                        data.append(self.byte_vocab[tok])
                    else:
                        # Add (token_id, token_start_idx, token_width) to results
                        data.append((self.byte_vocab[tok], i, len(tok)))
                    i += j  # advance to next token
                    break  # the for loop
        return data

    def tokenize_to_encoded_str(self, sent: Tokenizable) -> List[str]:
        """
        Return the tokenization as tokens in encoded str form
        """
        return [self.inv_str_vocab[token_id] for token_id in self.tokenize(sent, tokens_only=True)]

    def tokenize_to_bytes(self, sent: Tokenizable) -> List[bytes]:
        """
        Same as tokenize_to_encoded_strbut return byte form
        """
        return [self.inv_byte_vocab[token_id] for token_id in self.tokenize(sent, tokens_only=True)]

    @staticmethod
    def do_triples(combined, pad: int, padleft, padright, cur_id, sign, triples):
        """
        Add the appropriate (t,v,v') triples to our dictionary where t, v, and v' are all int indices.
        """
        # where the right padding starts
        right_ind = len(combined) - padright

        # iterate over the targets
        # note that the padding elements now have different contexts in
        # center section, so need to let them be targets too
        for t, target in enumerate(combined):
            # the contexts, need pad here not padleft or padright,
            # since some context may be within the combined
            for c in range(t - pad, t + pad + 1):
                # context is in range and distinct from target
                # ignore the case where both c and t are in padding since that cancels
                if c >= 0 and c != t and c < len(combined) and \
                        ((c >= padleft and c < right_ind) or (t >= padleft and t < right_ind)):
                    trip = (cur_id, target, combined[c])
                    # add sign to the triples
                    triples[trip] = triples.get(trip, 0) + sign

    def fast_sage(self, sent: Tokenizable, triples, ablated_sizes, pad: int=2, verbose: bool=False) -> int:
        """
        Tokenize the sentence `sent`, add to the counts in the triples dict tracking the (cur_id,t,c) for the ablated
        token cur_id, with target token t and context token c. Also updates the statistics in ablated_sizes.
        Returns the total_tokens from tokenizing `sent`.
        """
        n = len(sent)

        # returns triples of (ids, start_index, width)
        values = self.tokenize(sent)
        ids, start_indices, widths = zip(*values)
        # if you use np.array here, remember to fix concatenation below

        # note, these are arrays over the tokens so len(values) < n
        total_tokens = len(values)

        # tuples to list
        ids = list(ids)
        start_indices = list(start_indices)
        widths = list(widths)

        max_len = 0

        # have a constant time lookup on whether we're at a token on the base tokenization
        # if >= 0, is the index of the token in ids or widths
        on_base = np.zeros(n, dtype=int) - 1
        for j, si in enumerate(start_indices):
            on_base[si] = j
        # now we can just produce our ablated tokenizations
        # quite efficiently
        for loc, (cur_id, start_index, width) in enumerate(values):
            # skip single bytes
            if width > 1:

                ablated_tokenization = []

                # find the next token with width-1 or less
                # starting at start_index
                i = start_index
                for j in range(width - 1, 0, -1):
                    tok = sent[i:i + j]
                    if tok in self.byte_vocab:
                        ablated_tokenization.append(self.byte_vocab[tok])  # keep the ids
                        i += j  # advance to next token
                        break  # the for loop

                # now extend as normal until we get back on the old path
                while i < n:
                    for j in range(min(self.max_len, n - i), 0, -1):
                        tok = sent[i:i + j]
                        if tok in self.byte_vocab:
                            ablated_tokenization.append(self.byte_vocab[tok])
                            i += j  # advance to next token
                            break  # the for loop

                    # we never got back on the path, so set beyond to n
                    if i >= n:
                        beyond = n
                        break

                    # we get to a spot on the current longest path
                    # we're back to the old tokenization, set beyond accordingly
                    if on_base[i] != -1:
                        beyond = on_base[i]
                        break

                if verbose:
                    print(self.print_tokens(ablated_tokenization))

                # track how many tokens were required for the ablation
                lat = len(ablated_tokenization)
                ablated_sizes[lat] = ablated_sizes.get(lat, 0) + 1
                max_len = max(max_len, lat)

                # note: on_base[i] is one beyond the last difference
                base_tok = ids[loc:beyond]
                if verbose:
                    print(self.print_tokens(base_tok))

                # can we do any padding on left or right
                padleft = min(pad, loc)
                padright = min(pad, len(values) - beyond)
                left_pad = ids[loc - padleft:loc]
                # print(print_tokens(left_pad))
                # note: beyond is one beyond the last difference
                right_pad = ids[beyond:beyond + padright]
                # print(print_tokens(right_pad))

                # combine with the padding, and work out the context triples
                combined_ab = left_pad + ablated_tokenization + right_pad
                self.do_triples(combined_ab, pad, padleft, padright, cur_id, 1, triples)

                # and same for the base tokenization
                combined_base = left_pad + base_tok + right_pad
                self.do_triples(combined_base, pad, padleft, padright, cur_id, -1, triples)

                if verbose:
                    print("base:", self.print_tokens(left_pad), self.print_tokens(base_tok),
                          self.print_tokens(right_pad))
                    print("ab:  ", self.print_tokens(left_pad), self.print_tokens(ablated_tokenization),
                          self.print_tokens(right_pad))
                    print("comb base:", self.print_tokens(combined_base))
                    print("comb ab:", self.print_tokens(combined_ab))
                    print()

        # log some of these
        if max_len > 200:
            # remember to convert from bytes
            print("long max_len:", max_len, '"' + sent.decode('utf-8') + '"')

        return total_tokens


def verify_all_single_byte_exist_in_vocab(vocab):
    for i in range(256):
        b = bytes([i])
        if b not in vocab:
            raise Exception(f"missing byte {b}")
