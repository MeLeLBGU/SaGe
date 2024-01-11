import json
import pickle
import random
import sentencepiece as spm
import math
import numpy as np
import Utils
import multiprocessing as mp
import os

class SG_BPE_Models:
    def __init__(self, experiment_name, is_continue_execution, final_vocab_size, initial_vocab_size, partial_corpus_filepath):
        # SG-BPE model - this we are gonna update
        sg_bpe_model_prefix = "results/{}/sg_bpe".format(experiment_name)
        sg_bpe_model = sg_bpe_model_prefix + ".model"

        if not is_continue_execution:
            spm.SentencePieceTrainer.train(input=partial_corpus_filepath, model_prefix=sg_bpe_model_prefix, vocab_size=initial_vocab_size, model_type="bpe")
        self._sg_bpe_model = spm.SentencePieceProcessor(model_file=sg_bpe_model)

        # Vanilla BPE model - for comparisons
        vanilla_bpe_model_prefix = "results/{}/bpe_vanilla".format(experiment_name)
        vanilla_bpe_model = vanilla_bpe_model_prefix + ".model"
        
        if not is_continue_execution:
            spm.SentencePieceTrainer.train(input=partial_corpus_filepath, model_prefix=vanilla_bpe_model_prefix, vocab_size=final_vocab_size, model_type="bpe")
        self._bpe_vanilla_model = spm.SentencePieceProcessor(model_file=vanilla_bpe_model)

    def get_sg_bpe_model(self):
        return self._sg_bpe_model

    def get_bpe_vanilla_model(self):
        return self._bpe_vanilla_model


class Model:
    def __init__(self, experiment_name, log, model, model_name, \
        target_embeddings, context_embeddings, corpus_lines, max_lines_per_token, window_size, is_continue_execution=False, vocab_filepath=False):

        self._experiment_name = experiment_name
        self._model = model
        self._model_name = model_name
        self._target_embeddings = target_embeddings
        self._context_embeddings = context_embeddings
        self._log = log
        self._current_vocab = None
        self._should_compute_vocab = True
        self._corpus_lines = corpus_lines
        self._max_lines_per_token = max_lines_per_token
        self._window_size = window_size
              
        if is_continue_execution:
            with open(vocab_filepath + ".bin", "rb") as vocab_file:
                current_bpe_vocab = pickle.load(vocab_file)
                self.set_vocab(current_bpe_vocab)

    def initialize_encoded_form_for_corpus_lines(self):
        model_encoded_coprus_lines_token_ids = []
        model_encoded_coprus_lines_token_pieces = []
        for line in self._corpus_lines:
            tokens_in_line_ints = self._model.encode(line, out_type=int)
            tokens_in_line_pieces = [self._model.id_to_piece(x) for x in tokens_in_line_ints]

            model_encoded_coprus_lines_token_ids.append(tokens_in_line_ints)
            model_encoded_coprus_lines_token_pieces.append(tokens_in_line_pieces)

        self._model_encoded_corpus_lines_token_ids = model_encoded_coprus_lines_token_ids
        self._model_encoded_corpus_lines_token_pieces = model_encoded_coprus_lines_token_pieces

    def initialize_token_to_line_indices_dictionary(self, current_vocab, corpus_lines, experiment_name, is_continue_execution):
        token_to_line_indices_dict_filepath = "results/{}/token_to_line_indices_dict.bin".format(experiment_name)
        if is_continue_execution and os.path.exists(token_to_line_indices_dict_filepath):
            with open(token_to_line_indices_dict_filepath, "rb") as token_to_line_indices_dict_file:
                self._token_to_line_indices_dict = pickle.load(token_to_line_indices_dict_file)
        else:
            self._token_to_line_indices_dict = Utils.token_to_line_indices_dictionary(current_vocab, corpus_lines)
            with open(token_to_line_indices_dict_filepath, "wb") as token_to_line_indices_dict_file:
                pickle.dump(self._token_to_line_indices_dict, token_to_line_indices_dict_file)
        
    def update_encoded_form_for_corpus_lines(self, tokens_pruned):
        for token in tokens_pruned:
            line_indices_with_token = self._token_to_line_indices_dict[token]
            for index in line_indices_with_token:
                current_line = self._corpus_lines[index]
                self._model_encoded_corpus_lines_token_ids[index] = self._model.encode(current_line, out_type=int)
                self._model_encoded_corpus_lines_token_pieces[index] = [self._model.id_to_piece(x) for x in self._model_encoded_corpus_lines_token_ids[index]]

	# We use the "get_current_vocab" method because of sentencepiece tricky way to remove tokens from the vocabulary.
    def get_current_vocab(self):
        if not self._should_compute_vocab:
            return self._current_vocab

        model_vocab = []
        for i in range(self._model.vocab_size()):
            model_vocab.append(self._model.id_to_piece(i))

        return model_vocab

    def log_experiments_model_results(self, training_filepath, model_name_override=None):
        if model_name_override:
            model_name = model_name_override
        else:
            model_name = self._model_name
        
        ## Log model vocabulary
        model_vocab = self.get_current_vocab()
        model_vocab_filepath = ("./results/{}/" + model_name + "_vocab.txt").format(self._experiment_name)
        with open(model_vocab_filepath, "w+") as model_vocab_results_file:
            model_vocab_results_file.write(json.dumps(model_vocab, indent=4))

        ## Log encoding of input file with model vocabulary
        with open(training_filepath) as input_file:
            input_file_data = input_file.read()

        encoded_data = [self._model.id_to_piece(x) for x in self._model.encode(input_file_data)]
        model_encoding_filepath = ("./results/{}/" + model_name + "_encoding.txt").format(self._experiment_name)
        with open(model_encoding_filepath, "w+") as encoding_results_file:
            encoding_results_file.write(' '.join(encoded_data))

    def sg_for_window(self, target_token, window):
        current_p = 0
        for w in window:
            # calculate current value to add
            dot_product = np.dot(self._target_embeddings[target_token], self._context_embeddings[w])
            try:
                current_p += math.log(Utils.sigmoid(dot_product))
            except:
                pass

        return (-1) * current_p

    def token_context_sg_log_prob(self, token_int, i, tokens_in_line_ints):
        # get context window tokens
        window, _, _ = Utils.compute_window(i, tokens_in_line_ints, self._window_size)
        return self.sg_for_window(token_int, window)

    def total_sg_log_prob(self, training_filepath):
        # getting lines from corpus
        with open(training_filepath, "r") as training_file:
            corpus_lines = training_file.readlines()

        p = 0
        for line in corpus_lines:
            current_p = 0
            tokens_in_line_ints = self._model.encode(line, out_type=int)
            for i, token_int in enumerate(tokens_in_line_ints):
                current_p += self.token_context_sg_log_prob(token_int, i, tokens_in_line_ints)

            p += current_p

        return p

    # Compute total_sg_log_prob without each token
    # -- This is executed once in iteration --
    # Multi Processing version of this method ###########
    def get_sg_log_prob_without_tokens_mp(self, current_total_sg, training_filepath, nat_list=None, dict_of_top_tokens=None):
        token_and_sg_log_prob_without_it = {}
        current_vocab = self.get_current_vocab()

        if dict_of_top_tokens:
            current_vocab = dict_of_top_tokens

        print("\nCurrent vocab len - {}".format(len(current_vocab)))

        if nat_list:
            current_nat_list = Utils.get_not_ablateable_tokens_list(current_vocab)
            current_vocab = [t for t in current_vocab if t not in current_nat_list]

        process_pool = mp.Pool(mp.cpu_count())
        params = [(self._model, token, current_total_sg, \
                    current_vocab, training_filepath, \
                    self._target_embeddings, self._context_embeddings, self._log, self._corpus_lines, self._window_size) for token in current_vocab]
        res = process_pool.starmap(Utils.sg_wo_token_mp, params)
        
        for r in res:
            token = r[0]
            current_sg_log_prob = r[1]
            token_and_sg_log_prob_without_it[token] = current_sg_log_prob

        return token_and_sg_log_prob_without_it

    # More optimized form of "get_sg_log_prob_without_tokens_mp"
    def get_sg_log_prob_without_tokens_mp2(self, current_total_sg, nat_list=None, dict_of_top_tokens=None):
        token_and_sg_log_prob_without_it = {}
        
        current_vocab = self.get_current_vocab()
        if dict_of_top_tokens:
            current_vocab = dict_of_top_tokens

        if nat_list:
            current_nat_list = Utils.get_not_ablateable_tokens_list(current_vocab)
            current_vocab = [t for t in current_vocab if t not in current_nat_list]

        process_pool = mp.Pool(mp.cpu_count())
        lines_per_token = {}

        params = []
        for token in current_vocab:
            lines_to_consider = self._token_to_line_indices_dict[token]
            if len(lines_to_consider) > self._max_lines_per_token:
                lines_to_consider = random.sample(lines_to_consider, self._max_lines_per_token)

            for line_index in lines_to_consider:
                params.append((self._model, line_index, \
                    self._corpus_lines, self._model_encoded_corpus_lines_token_ids, self._model_encoded_corpus_lines_token_pieces, \
                    token, current_vocab, \
                    self._target_embeddings, self._context_embeddings, self._log, self._window_size))

            lines_per_token[token] = len(lines_to_consider)
        
        res = process_pool.starmap(Utils.get_diff_sg_wo_token_for_line, params)

        for r in res:
            token = r[0]
            sg_wo_diff = r[1]

            if token not in token_and_sg_log_prob_without_it.keys():
                token_and_sg_log_prob_without_it[token] = current_total_sg

            token_and_sg_log_prob_without_it[token] += (float(sg_wo_diff) / lines_per_token[token])
            
        return token_and_sg_log_prob_without_it

    def get_model(self):
        return self._model

    def set_vocab(self, new_vocabulary):
        # Note: this indeed changing the model vocab, but not model.vocab_size().
        # if you want to test this - remove some token from the vocabulary, and compare encodings before and after removing.
        self._model.set_vocabulary(new_vocabulary)
        self._current_vocab = new_vocabulary
        self._should_compute_vocab = False
        
