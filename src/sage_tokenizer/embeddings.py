# Copyright © 2023 Kensho Technologies, LLC

import time
import logging

import gensim.models
import numpy as np

from typing import List, Iterable
from pathlib import Path

from .Word2VecParams import Word2VecParams
from .model import SaGeTokenizer
from .paths import getDataFolder
from .utils import FileAsStringIterable


def get_embeddings(vocab_size: int, embeddings_folder: Path, partial_corpus: Iterable[str], sage_model: SaGeTokenizer, workers_number: int, word2vec_params: Word2VecParams) -> np.ndarray:
    logging.info(f"training Embeddings at vocab size {vocab_size}")
    embeddings_folder = Path(embeddings_folder)

    # is there an embedding of this size
    embeddings_filepath = embeddings_folder / f"embeddings_{vocab_size}.npy"
    if embeddings_filepath.exists():
        logging.info(f"Found trained embeddings. Loading it from {embeddings_filepath.as_posix()}")
        # context and target embeddings are the same so just keep one copy around
        embeddings = np.load(embeddings_filepath.as_posix())
    else:
        logging.info(f"Start training embeddings with Word2Vec...")
        start_time = time.time()
        embeddings = train_embeddings(sage_model, partial_corpus, workers_number, word2vec_params, embeddings_folder)
        logging.info(f"Embeddings time: {time.time() - start_time}")
        logging.info(f"Save embeddings to {embeddings_filepath.as_posix()}")
        np.save(embeddings_filepath.as_posix(), embeddings, allow_pickle=True)
    return embeddings


def train_embeddings(sage_model: SaGeTokenizer, partial_corpus: Iterable[str], workers: int, word2vec_params: Word2VecParams, embeddings_folder: Path) -> np.ndarray:

    def tokenisedCorpus() -> Iterable[str]:
        start = time.time()
        logging.info(f"Tokenizing corpus...")
        for i,s in enumerate(partial_corpus):
            if i % 1_000_000 == 0:
                logging.info(f"\tTokenizing example {i}, time: {(time.time() - start):.2f} seconds")
            yield " ".join(sage_model.tokenize_to_encoded_str(bytes(s, 'utf-8')))  # GenSim expects a corpus consisting of whitespace-separated tokens.

    if isinstance(partial_corpus, FileAsStringIterable):  # GenSim is accelerated for file-stored corpora (https://github.com/RaRe-Technologies/gensim/releases/tag/3.6.0 and https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Any2Vec_Filebased.ipynb).
        gensim_file = embeddings_folder / f"gensim_{sage_model.vocab_size()}.txt"
        if gensim_file.exists():  # Caching
            logging.info(f"Tokenized corpus already exists at {gensim_file.as_posix()}")
        else:
            with open(gensim_file, "w", encoding="utf-8") as handle:
                for token_string in tokenisedCorpus():
                    handle.write(token_string + "\n")
            logging.info(f"Tokenized data written at {gensim_file.as_posix()}")

        gensim_iterator = None
        gensim_file = gensim_file.as_posix()
    else:
        gensim_iterator = tokenisedCorpus()
        gensim_file = None

    word2vec_model = gensim.models.Word2Vec(corpus_file=gensim_file,
                                            sentences=gensim_iterator,

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
