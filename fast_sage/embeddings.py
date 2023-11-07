# Copyright © 2023 Kensho Technologies, LLC

import gensim.models
import numpy as np
import os, time

import utils
from params_config import *

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

def train_embeddings(sage_model, partial_corpus, workers):
    # sentences = CorpusIterator(model, corpus_filepath)
    # parameters are set in params_config

    # also save in a version of this with a sentance per line, whitespace per token
    # which gensim's word2vec wants to process things in parallel
    # see https://github.com/RaRe-Technologies/gensim/releases/tag/3.6.0
    # and https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Any2Vec_Filebased.ipynb
    gensim_corpus_filepath = f"data/gensim_{sage_model.vocab_size()}.txt"

    # if exists after the first time then skip
    if os.path.exists(gensim_corpus_filepath):
        print(f"Gensim format data file already exists: {gensim_corpus_filepath}")
    else:
        gensim_start = time.time()
        print(f"starting tokenization of {len(partial_corpus)} lines for gensim")
        with open(gensim_corpus_filepath, "wt") as out:
            for i, line in enumerate(partial_corpus):
                if i % 1000000 == 0:
                    print(i, time.time() - gensim_start)
                out.write(" ".join(sage_model.tokenize_to_encoded_str(bytes(line, 'utf-8'))) + "\n")
        print(f"Gensim format data written: {gensim_corpus_filepath}, time:{time.time()-gensim_start}")

    word2vec_model = gensim.models.Word2Vec(corpus_file=gensim_corpus_filepath,
                                            vector_size=config.D,
                                            negative=config.N,
                                            alpha=config.ALPHA,
                                            window=config.window_size,
                                            min_count=config.min_count,
                                            sg=config.sg,
                                            workers=workers)

    embeddings = np.zeros(shape=(sage_model.vocab_size(), config.D))

    for idx, token in sage_model.inv_str_vocab.items():
        if token in word2vec_model.wv.key_to_index.keys():
            embeddings[idx] = word2vec_model.wv[token]
        else:
            # some may not have made the min_count value, so will be missing
            # Embeddings not found for this token. Assign a random vector
            # doing this the same way as the old SaGe code
            embeddings[idx] = np.random.uniform(low=-0.5/config.D, high=0.5/config.D, size=(1, config.D))

    # just return one copy that we'll use for both context and target embeddings
    return embeddings
