import Utils
import random

class Corpus:
    def __init__(self, corpus_filepath, partial_corpus_filepath, partial_corpus_lines_number, log):
        self._log = log

        with open(corpus_filepath) as full_corpus:
            self._full_corpus_data = full_corpus.readlines()
            self._log.info("original corpus num of lines: {}".format(len(self._full_corpus_data)))
            random.shuffle(self._full_corpus_data)
            self._partial_corpus_data = self._full_corpus_data[:partial_corpus_lines_number]
            self._log.info("num of lines in corpus: {}".format(len(self._partial_corpus_data)))

        with open(partial_corpus_filepath, "w+") as partial_corpus_file:
            partial_corpus_file.writelines(self._partial_corpus_data)

    def get_full_corpus(self):
        return self._full_corpus_data

    def get_partial_corpus(self):
        return self._partial_corpus_data

    def get_corpus(self, partial=True):
        if partial:
            cor = self.get_partial_corpus()
        else:
            cor = self.get_full_corpus()

        self._log.info("corpus num of lines: {}".format(len(cor)))
        return cor

    def compute_window(self, token_index, tokens_in_line):
        return Utils.compute_window(token_index, tokens_in_line)

