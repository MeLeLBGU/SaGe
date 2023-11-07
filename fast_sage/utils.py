# Copyright © 2023 Kensho Technologies, LLC

import os
import time
import random
import numpy as np
# import logging
from scipy.special import expit

# only log code outside of multiprocessing
# logger = logging.getLogger(__name__)

# map all bytes to valid utf-8 characters
# in the same way that the huggingface tokenizers byte level pretokenizer does
class HFEncoding:

    # translated from the rust code
    # see https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs
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
    def tobytes(self, s : str) -> bytes:
        return b"".join([self.inv_byte_map[c] for c in s])

    # convert a byte string into an encoded string of valid characters
    def toencoded(self, byte_str : bytes) -> str:
        return "".join([self.byte_map[bytes([c])] for c in byte_str])

    # TODO: write tests for this

# dump the vocab to a file, encoded as characters here
# no special tokens are added
# are saved in same order by index, so should preserve order
def write_vocab(vocab, filename):
    vocab_size = len(vocab)

    # write these in increasing index order
    # so same as any previous order
    byindex = sorted([(idx,token) for token,idx in vocab.items()])

    with open(filename, 'w') as f:
        for _, token in byindex:
            f.write(token.hex() + '\n')


def write_sorted_losses(sl, filename, sage_model):
    with open(filename, 'w') as f:
        for loss, tid in sl:
            f.write(sage_model.id_to_encoded(tid) + "\t" + str(loss) + "\n")

# read our hex formatted vocab file
# return a list of bytes objects
# input file has one vocab word per line,
# each hex encoded
def load_vocab(vocab_filepath):

    if not os.path.exists(vocab_filepath):
        raise FileNotFoundError(f'Missing vocab file: {vocab_filepath}')

    with open(vocab_filepath) as vocab_file:
        # fromhex ignores whitespace from \n at end
        initial_vocab = [bytes.fromhex(token) for token in vocab_file.readlines()]

    return initial_vocab


def load_corpus(sage_model, corpus_filepath, partial_corpus_filepath, partial_corpus_lines_number):

    if not os.path.exists("data"):
        print("'data/' not found. Creating one for storing partial corpus...")
        os.mkdir("data")

    if os.path.exists(partial_corpus_filepath):
        # corpus already exists, directly loading
        print(f"Found Processed Partial Corpus. Directly Loading from {partial_corpus_filepath}...")
        read_start = time.time()
        with open(partial_corpus_filepath) as corpus_f:
            partial_corpus = corpus_f.readlines()
        print("Size of Corpus", len(partial_corpus), time.time()-read_start)
    else:
        read_start = time.time()
        with open(corpus_filepath) as full_corpus_f:
            corpus = full_corpus_f.readlines()
            print(f"Loading from Original Corpus. Number of lines: {len(corpus)}")

        random.shuffle(corpus)
        print("Original Corpus read and shuffled", time.time() - read_start)

        # may be same as original depending on partial_corpus_lines_number
        write_start = time.time()
        partial_corpus = corpus[:partial_corpus_lines_number*1000]
        corpus_filename = os.path.splitext(os.path.basename(corpus_filepath))[0]
        partial_corpus_filepath = f"data/{corpus_filename}_{len(partial_corpus)}.txt"
        with open(partial_corpus_filepath, "w+") as partial_corpus_f:
             partial_corpus_f.writelines(partial_corpus)
        print(f"Partial corpus saved at {partial_corpus_filepath}. Number of lines: {len(partial_corpus)}, time:{time.time()-write_start}")

    return partial_corpus, partial_corpus_filepath

# Split the data given the number of chunks we expect
# Returns a generator
def divide_data_by_num(data, num_procs):
    size_per_chunk = len(data) // num_procs
    for i in range(0, len(data), size_per_chunk+1):
        yield data[i: i+size_per_chunk+1]

# Split the data given the size of chunks we expect
# Returns a generator
def divide_data_by_size(data, size):
    for i in range(0, len(data), size):
        yield data[i: i+size]

def fix_random_seed(experiment_name, random_seed):
    seed_filepath = f"results/{experiment_name}/seed.txt"
    with open(seed_filepath, "w+") as f:
        f.write(str(random_seed))
    random.seed(random_seed)
    np.random.seed(random_seed)


def verify_all_bytes(vocab):
    for i in range(256):
        b = bytes([i])
        if b not in vocab:
            print("missing byte", b)
        assert b in vocab

# function for computing losses given triple counts and embeddings
# losses : accumulate losses per ablated token, excluding the single byte ones, side effect this
# all_triples : triple values to aggregate into losses
# embeddings : embedding for each token
def compute_losses(tid, row, losses, all_triples, embeddings):
    for idx, ((ablated_token_id, target_id, context_id), count) in enumerate(all_triples.items()):
        product = np.log(expit(np.dot(embeddings[target_id], embeddings[context_id])))
        losses[ablated_token_id] = losses.get(ablated_token_id, 0.0) + count * product

# function that runs sage on each chunk of data (in parallelization)
# note: this is called from multiprocessing, so use print rather than logging
def sage_per_chunk(tid, model, data, embeddings, chunksize = 10000):

    print(f"Starting chunk {tid}, with {len(data)} lines of data")

    start_time = time.time()

    # accumulate over all the data
    losses = {}

    # these accumulate over each size
    triples = {}
    ablated_sizes = {}
    total_tokens = 0
    total_triples = 0
    total_fs_time = 0.0
    total_cl_time = 0.0

    fs_start = time.time()
    for row, d in enumerate(data):

        total_tokens += model.fast_sage(bytes(d, 'utf-8'), triples, ablated_sizes)

        # if filled up chunk, then compute the losses
        # to free up memory
        if (row > 0) and (row % chunksize == 0):

            # take the total time here over all calls
            fs_time = time.time() - fs_start
            total_fs_time += fs_time
            # reinitialize fs_start
            fs_start = time.time()

            cl_start = time.time()
            compute_losses(tid, row, losses, triples, embeddings)
            cl_time = time.time() - cl_start
            total_cl_time += cl_time

            print(f"fast_sage {tid}, row {row} of {len(data)}, fs_time: {fs_time}, cl_time:{cl_time}, triples:{len(triples)}, tokens:{total_tokens}")

            # total these up
            total_triples += len(triples)

            # zero out the triples from this chunksize lines
            triples = {}

    # compute for final partial chunk
    compute_losses(tid, row, losses, triples, embeddings)

    # add final batch
    total_triples += len(triples)

    # the triples can get quite large, so to avoid merging these
    # dict values, let's compute the losses in parallel too
    print(f"final fast_sage {tid}, row {row} of {len(data)}, fs_time: {total_fs_time}, cl_time:{total_cl_time}, time:{time.time()-start_time}, triples:{len(triples)}, tokens:{total_tokens}")

    # Extra negative sign for equation (1) in SaGe paper
    # track number in cache too
    losses = {k: -v for k, v in losses.items()}

    return losses, total_tokens, total_triples, ablated_sizes
