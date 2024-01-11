import numpy as np
import Parameters
import gensim.models

############################################################################

class CorpusIteratorForGensim:
    def __init__(self, corpus, model):
        self._corpus = corpus
        self._model = model

    def __iter__(self):
        corpus_lines = self._corpus.get_corpus()

        for line in corpus_lines:
            # yield tokenized line, when representing tokens as strings (not ints)
            tokens_in_line = [self._model.id_to_piece(x) for x in self._model.encode(line, out_type=int)]
            yield tokens_in_line

############################################################################

class EmbeddingsTrainer:
    def __init__(self, model, corpus, window_size, log):
        self._model = model
        self._corpus = corpus
        self._window_size = window_size
        self._log = log

    ######################################

    def construct_word2vec_embeddings(self, word2vec_model):
        vocab_size = self._model.vocab_size()
        embeddings = np.zeros(shape=(vocab_size, Parameters.D))
        
        for i in range(vocab_size):
            ith_token = self._model.id_to_piece(i)
            if ith_token in word2vec_model.wv.key_to_index.keys():    
                embeddings[i] = word2vec_model.wv[ith_token]
            else:
                self._log.warning("No wv for token {}. Assigning random vector...".format(ith_token))
                embeddings[i] = np.random.uniform(low=-0.5 / Parameters.D, high=0.5 / Parameters.D, size=(1, Parameters.D))

        return embeddings

    ######################################

    def train_embeddings(self):
        self._log.info("Training embeddings.")
        sentences = CorpusIteratorForGensim(self._corpus, self._model)
        self._log.info("Built CorpusIteratorForGensim")
        word2vec_model = gensim.models.Word2Vec(sentences=sentences, \
                                                vector_size=Parameters.D, \
                                                window=self._window_size,
                                                min_count=0,
                                                sg=1,
                                                negative=Parameters.N)
        self._log.info("gensim.models.Word2Vec Finished")

        # now extract the embeddings in the format we expect to
        embeddings = self.construct_word2vec_embeddings(word2vec_model)
        self._log.info("Finished construct_word2vec_embeddings")

        # in gensim there is no way to get context embeddings - so we do target=context=word vectors.
        return embeddings, embeddings

############################################################################
