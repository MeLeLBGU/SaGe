import numpy as np
import Utils
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

class UnigramTable:
  # Table used for negative sampling.
  # For now - not following the power law distribution, but uniform distribution.
  def __init__(self, bpe_model):
    self.negative_samples_table = []
    for i in range(bpe_model.vocab_size()):
      ith_token = bpe_model.id_to_piece(i)
      self.negative_samples_table.append(ith_token)

  def sample(self, count):
    indices = np.random.randint(low=0, 
                                high=len(self.negative_samples_table),
                                size = count)
    return [self.negative_samples_table[i] for i in indices]

############################################################################

class WordpieceTrainer:
    def __init__(self, model, corpus, window_size, log, use_gensim=False):
        self._model = model
        self._corpus = corpus
        self._window_size = window_size
        self._use_gensim = use_gensim
        self._log = log

        if not use_gensim:
            self._negative_samples_table = UnigramTable(self._model)

    ######################################

    def init_net(self, d):
        vocab_size = self._model.vocab_size()

        # Init target_embeddings with random numbers from a uniform distribution
        #   on the interval [-0.5/d, 0.5/d]
        target_embeddings = np.random.uniform(low=-0.5/d, high=0.5/d, size=(vocab_size, d))

        # Init context_embeddings with zeros (as in original word2vecf code)
        context_embeddings = np.zeros(shape=(vocab_size, d))

        return target_embeddings, context_embeddings

    ######################################

    def train_for_positive_context_in_line(self,
                                       tokens_in_line, 
                                       i, 
                                       positive_context,
                                       target_embeddings,
                                       context_embeddings):

        window, context_start, context_end = Utils.compute_window(i, tokens_in_line, self._window_size)

        for target_token in window:
            neu1e = np.zeros(Parameters.D)

            negative_samples = self._negative_samples_table.sample(Parameters.N)
            samples = [(positive_context, 1)] + \
                      [(self._model.piece_to_id(n), 0) for n in negative_samples]

            for context, label in samples:
                z = np.dot(target_embeddings[target_token], context_embeddings[context])
                p = Utils.sigmoid(z)
                g = Parameters.ALPHA * (label - p)
                neu1e += g * context_embeddings[context]
                context_embeddings[context] += g * target_embeddings[target_token]

            target_embeddings[target_token] += neu1e

        return target_embeddings, context_embeddings

    ######################################

    def train_embeddings_not_gensim(self):
        self._log.info("In train_embeddings_not_gensim")
        target_embeddings, context_embeddings = self.init_net(Parameters.D)

        corpus = self._corpus.get_corpus()

        for line in corpus:
            tokens_in_line = self._model.encode(line, out_type=int)
            for i, positive_context in enumerate(tokens_in_line):
                target_embeddings, context_embeddings = \
                    self.train_for_positive_context_in_line(tokens_in_line,
                                                            i, 
                                                            positive_context,
                                                            target_embeddings,
                                                            context_embeddings)
          
        return target_embeddings, context_embeddings

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
                embeddings[i] = np.random.uniform(low=-0.5/Parameters.D, high=0.5/Parameters.D, size=(1, Parameters.D))

        return embeddings

    ######################################

    def train_embeddings_gensim(self):
        self._log.info("In train_embeddings_gensim")
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

    ######################################

    def train_embeddings(self):
        self._log.info("Training embeddings.")

        if not self._use_gensim:
            return self.train_embeddings_not_gensim()

        return self.train_embeddings_gensim()

############################################################################

