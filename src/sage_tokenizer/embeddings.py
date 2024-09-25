# Copyright © 2023 Kensho Technologies, LLC

import time
import logging

import gensim.models
import numpy as np

from typing import List
from pathlib import Path

# class CorpusIterator:
#     def __init__(self, model, corpus_filepath):
#         self._model = model
#         self.corpus_filepath = corpus_filepath

#     def __iter__(self):
#         with open(self.corpus_filepath) as f:
#             corpus_data = f.readlines()
#         for line in corpus_data:
#             # convert bytes to tokens in encoded string form, for gensim
#             yield self._model.tokenize_to_encoded_str(bytes(line, 'utf-8'))
from .Word2VecParams import Word2VecParams
from .model import SaGeTokenizer
from .paths import getDataFolder


def get_embeddings(vocab_size: int, embeddings_folder: Path, partial_corpus: List[str], sage_model: SaGeTokenizer, workers_number: int, word2vec_params: Word2VecParams) -> np.ndarray:
    logging.info(f"training Embeddings at vocab size {vocab_size}")
    # is there an embedding of this size
    embeddings_filepath = Path(embeddings_folder) / f"embeddings_{vocab_size}.npy"
    if embeddings_filepath.exists():
        logging.info(f"Found trained embeddings. Loading it from {embeddings_filepath.as_posix()}")
        # context and target embeddings are the same so just keep one copy around
        embeddings = np.load(embeddings_filepath.as_posix())
    else:
        logging.info(f"Start training embeddings with Word2Vec...")
        start_time = time.time()
        embeddings = train_embeddings(sage_model, partial_corpus, workers_number, word2vec_params)
        logging.info(f"Embeddings time: {time.time() - start_time}")
        logging.info(f"Save embeddings to {embeddings_filepath.as_posix()}")
        np.save(embeddings_filepath.as_posix(), embeddings, allow_pickle=True)
    return embeddings


def train_embeddings(sage_model: SaGeTokenizer, partial_corpus: List[str], workers: int, word2vec_params: Word2VecParams) -> np.ndarray:
    # sentences = CorpusIterator(model, corpus_filepath)

    # also save in a version of this with a sentence per line, whitespace per token
    # which gensim's word2vec wants to process things in parallel
    # see https://github.com/RaRe-Technologies/gensim/releases/tag/3.6.0
    # and https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Any2Vec_Filebased.ipynb
    gensim_corpus_filepath = getDataFolder() / f"gensim_{sage_model.vocab_size()}.txt"

    # if already exists, use otherwise create the file
    if gensim_corpus_filepath.exists():
        logging.info(f"Gensim format data file already exists: {gensim_corpus_filepath.as_posix()}")
    else:
        gensim_start = time.time()
        logging.info(f"starting tokenization of {len(partial_corpus)} lines for gensim")
        with open(gensim_corpus_filepath, "w", encoding="utf-8") as gensim_file:
            for i, line in enumerate(partial_corpus):
                if i % 1_000_000 == 0:
                    logging.info(f"tokenizing line {i}, time: {(time.time() - gensim_start):.2f}")
                gensim_file.write(" ".join(sage_model.tokenize_to_encoded_str(bytes(line, 'utf-8'))) + "\n")
        logging.info(f"Gensim format data written: {gensim_corpus_filepath.as_posix()}, time: {(time.time()-gensim_start):.2f}")

    word2vec_model = gensim.models.Word2Vec(corpus_file=gensim_corpus_filepath.as_posix(),
                                            vector_size=word2vec_params.D,
                                            negative=word2vec_params.N,
                                            alpha=word2vec_params.ALPHA,
                                            window=word2vec_params.window_size,
                                            min_count=word2vec_params.min_count,
                                            sg=word2vec_params.sg,
                                            workers=workers)

    embeddings = np.zeros(shape=(sage_model.vocab_size(), word2vec_params.D))

    for idx, token in sage_model.inv_str_vocab.items():
        if token in word2vec_model.wv.key_to_index.keys():
            embeddings[idx] = word2vec_model.wv[token]
        else:
            # some may not have made the min_count value, so will be missing
            # Embeddings not found for this token. Assign a random vector
            # doing this the same way as the old SaGe code
            embeddings[idx] = np.random.uniform(low=-0.5/word2vec_params.D, high=0.5/word2vec_params.D, size=(1, word2vec_params.D))

    # just return one copy that we'll use for both context and target embeddings
    return embeddings
