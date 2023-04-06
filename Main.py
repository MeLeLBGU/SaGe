# -*- coding: utf-8 -*-

from functools import partial
import multiprocessing as mp
from pickle import FALSE
import sys

sys.path.insert(0, "./Python-Modules")

import math
import json
import random
import string
import pickle
import argparse
import Utils # From Python-Modules
import Corpus # From Python-Modules
import Logger # From Python-Modules
import SG_BPE # From Python-Modules
import Wordpiece # From Python-Modules
import numpy as np
from tqdm import tqdm
import sentencepiece as spm
from os.path import exists

######### Fix Random Seed ###########################################################

def set_random_seed(chosen_seed):
	# setting seed
	random.seed(chosen_seed)
	spm.SetRandomGeneratorSeed(chosen_seed)
	np.random.seed(chosen_seed)
	#print("random state: {}".format(random.getstate()))

def fix_random_seed(experiment_name, is_continue_execution, log2):
	chosen_seed = 0
	seed_filepath = "results/{}/seed.txt".format(experiment_name)
	if is_continue_execution and exists(seed_filepath):
		with open(seed_filepath, "r") as seed_file:
			chosen_seed = int(seed_file.read())
			log2.info("chosen seed is {}".format(chosen_seed))
	
	else:
		# saving for continue execution of that experiment, if this is first time execution
		with open(seed_filepath, "w+") as seed_file:
			seed_file.write(str(chosen_seed))

	set_random_seed(chosen_seed)
	
########## Uncomment and call to test token_to_line counters #######################

'''def token_to_line_counters(log2):
	sg_bpe_550_vocab_filepath = "results/exp_20/current_vocab.bin"
	with open(sg_bpe_550_vocab_filepath, "rb") as vocab_file:
		current_bpe_vocab = pickle.load(vocab_file)

	token_to_indices_dict = Utils.token_to_line_indices_dictionary(current_bpe_vocab, partial_corpus)
	
	token_to_indices_count_dict = {}
	for t in token_to_indices_dict.keys():
		token_to_indices_count_dict[t] = len(token_to_indices_dict[t])

	sorted_token_to_indices_count_dict = sorted(token_to_indices_count_dict.items(), key=lambda item: item[1])
	log2.info(json.dumps(sorted_token_to_indices_count_dict, indent=4))

	raise -1'''

######### Main #####################################################################

def log_parameters(log2, final_vocab_size, initial_vocab_size, \
	partial_corpus_lines_number, tokens_to_prune_in_iteration, tokens_to_consider_in_iteration, \
	iterations_until_reranking, max_lines_per_token, corpus_filepath, partial_corpus_filepath, window_size, use_gensim):
	
	log2.info("-----------------------------------------")
	log2.info("Starting Experiment.")
	log2.info("initial_vocab_size: {}, final_vocab_size: {}".format(initial_vocab_size, final_vocab_size))
	log2.info("partial_corpus_lines_number: {}\n".format(partial_corpus_lines_number))
	log2.info("tokens_to_prune_in_iteration {}".format(tokens_to_prune_in_iteration))
	log2.info("tokens_to_consider_in_iteration: {}".format(tokens_to_consider_in_iteration))
	log2.info("iterations_until_reranking: {}".format(iterations_until_reranking))
	log2.info("max_lines_per_token: {}".format(max_lines_per_token))
	log2.info("corpus_filepath: {}".format(corpus_filepath))
	log2.info("partial_corpus_filepath: {}".format(partial_corpus_filepath))
	log2.info("window_size: {}".format(window_size))
	log2.info("use_gensim: {}".format(use_gensim))
	log2.info("Number of processes in pool: {}".format(mp.cpu_count()))
	log2.info("Not re-calculating embeddings")
	log2.info("-----------------------------------------")

######### Main #####################################################################

def main(experiment_name, is_continue_execution, final_vocab_size, \
	initial_vocab_size, partial_corpus_lines_number, tokens_to_prune_in_iteration, tokens_to_consider_in_iteration, \
	iterations_until_reranking, max_lines_per_token, corpus_filepath, partial_corpus_filepath, window_size, use_gensim):

	# Initialize Statistics Logger
	print("Initializing statistics logger")
	log2 = Logger.Logger2("statistics")

	log2.info("Fixing random seed")
	fix_random_seed(experiment_name, is_continue_execution, log2)

	# Preparing Wiki 2m Data
	# Corpus currently returns partial data, shuffled
	log2.info("Preparing corpus")
	corpus = Corpus.Corpus(corpus_filepath, partial_corpus_filepath, partial_corpus_lines_number, log2)
	partial_corpus = corpus.get_corpus()

	# Log parameters if not continuation execution
	if not is_continue_execution:
		log_parameters(log2, final_vocab_size, initial_vocab_size, partial_corpus_lines_number, \
			tokens_to_prune_in_iteration, tokens_to_consider_in_iteration, iterations_until_reranking, \
			max_lines_per_token, corpus_filepath, partial_corpus_filepath, window_size, use_gensim)

	# BPE using SentencePiece Python Module
	log2.info("Preparing SG and BPE Models")
	models = SG_BPE.SG_BPE_Models(experiment_name, is_continue_execution, final_vocab_size, initial_vocab_size, partial_corpus_filepath)
	sg_bpe_model = models.get_sg_bpe_model()
	bpe_vanilla_500_model = models.get_bpe_vanilla_model()

	# Computing WordPiece Embedding Matrix
	log2.info("Computing embeddings")
	embeddings_filepath = "results/{}/embeddings.bin".format(experiment_name)
	if not is_continue_execution:
		wp_trainer = Wordpiece.WordpieceTrainer(sg_bpe_model, corpus, window_size, log2, use_gensim)
		target_embeddings, context_embeddings = wp_trainer.train_embeddings()
		with open(embeddings_filepath, "wb") as embeddings_file:
			pickle.dump(target_embeddings, embeddings_file)
			pickle.dump(context_embeddings, embeddings_file)
			log2.info("dumped embeddings to {}".format(embeddings_filepath))
	else:
		with open(embeddings_filepath, "rb") as embeddings_file:
			target_embeddings = pickle.load(embeddings_file)
			context_embeddings = pickle.load(embeddings_file)

	# Creating model objects
	# These will hold needed logic for training loop, logging results and gathering information about models too
	log2.info("Creating model objects")
	vocab_filepath = "results/{}/current_vocab".format(experiment_name)
	sg_bpe_model_object = SG_BPE.Model(experiment_name, log2, sg_bpe_model, "sg_bpe_original_550", \
		target_embeddings, context_embeddings, partial_corpus, max_lines_per_token, window_size, is_continue_execution, vocab_filepath)
	bpe_vanilla_model_object = SG_BPE.Model(experiment_name, log2, bpe_vanilla_500_model, "bpe_vanilla_500", \
		target_embeddings, context_embeddings, partial_corpus, max_lines_per_token, window_size)

	# We use the "get_current_vocab" method because of sentencepiece tricky way to remove tokens from the vocabulary.
	current_vocab = sg_bpe_model_object.get_current_vocab()
	with open(vocab_filepath + ".bin", "wb") as vocab_file:
		pickle.dump(current_vocab, vocab_file)

	###########################################################################################################################
	# Prune Tokens - SkipGram Log Probability
	# For now we compute SG objective without negative samples - this causes noise, and due to the fact we look at the actual result 
	# 	(and don't just refer it as objective) we don't want that noise.
	# We say we have better vocabulary when we minimize this objective.
	# Thus, in each iteration we will find the tokens that without them the objective is minimal.
	###########################################################################################################################

	log2.info("Preparing sorted Tokens-SG-objective list")
	sorted_tokens_sg_filepath = "results/{}/sorted_tokens_sg.bin".format(experiment_name)
	if is_continue_execution and exists(sorted_tokens_sg_filepath):
		with open(sorted_tokens_sg_filepath, "rb") as sorted_tokens_sg_file:
			sorted_tokens_sg = pickle.load(sorted_tokens_sg_file)
	else:
		# Log starting-point vocabulary:	
		sg_bpe_model_object.log_experiments_model_results(partial_corpus_filepath)
		bpe_vanilla_model_object.log_experiments_model_results(partial_corpus_filepath)

		# And Log starting-point SG objective
		sg_bpe_model_object.total_sg_log_prob(partial_corpus_filepath)
		log2.info("Initial SG-BPE-500 total_log_sg_prob: {}".format(total_skipgram_ns_probability))

		bpe_vanilla_model_object.total_sg_log_prob(partial_corpus_filepath)
		log2.info("Initial Vanilla-BPE-500 total_log_sg_prob: {}".format(bpe_vanilla_total_skipgram_ns_probability))

		log2.log_separator()

		# Log original sg log prob without each token
		## You can comment it - we used it to choose the hyperparameter of how many tokens we need to prune in each iteration.
		original_token_and_sg_log_prob_without_it = sg_bpe_model_object.get_sg_log_prob_without_tokens_mp(total_skipgram_ns_probability, partial_corpus_filepath, True)
		sorted_tokens_sg = sorted(original_token_and_sg_log_prob_without_it.items(), key=lambda item: item[1])
		sg_log_prob_without_tokens_mp_filepath = "./results/{}/sg_log_probs_without_tokens_mp.txt".format(experiment_name)
		with open(sg_log_prob_without_tokens_mp_filepath, "w+") as logprob_file:
			logprob_file.write(json.dumps(sorted_tokens_sg, indent=4))

		with open(sorted_tokens_sg_filepath, "wb") as sorted_tokens_sg_file:
			pickle.dump(sorted_tokens_sg, sorted_tokens_sg_file)

	# And start the "Training-Loop"
	log2.info("Starting the 'Training-Loop'")

	# sorted tokens list
	list_of_sorted_tokens_sg = [x[0] for x in sorted_tokens_sg]

	# initial vocab list
	current_vocab = sg_bpe_model_object.get_current_vocab()
	current_total_sg = total_skipgram_ns_probability # We already computed sg_bpe_model_object.total_sg_log_prob(partial_corpus_filepath)

	## prepare from ahead ####
	### now we want dict from token to index of lines - and re-encode when token is ablated
	sg_bpe_model_object.initialize_encoded_form_for_corpus_lines()
	sg_bpe_model_object.initialize_token_to_line_indices_dictionary(current_vocab, partial_corpus, experiment_name, is_continue_execution)

	# loop and ablate
	iteration = 0
	current_dict_of_top_tokens = {}

	progress_bar = tqdm(total = (len(current_vocab) - final_vocab_size) / tokens_to_prune_in_iteration)
	while len(current_vocab) > final_vocab_size:
		log2.info("iteration #{}, num of vocab: {}, length of top: {}".format(iteration, len(current_vocab), len(current_dict_of_top_tokens)))

		if (iteration % iterations_until_reranking) == 0:
			token_and_sg_log_prob_without_it = sg_bpe_model_object.get_sg_log_prob_without_tokens_mp2(current_total_sg, True)
			sorted_tokens_sg = sorted(token_and_sg_log_prob_without_it.items(), key=lambda item: item[1])
			current_dict_of_top_tokens = [t[0] for t in sorted_tokens_sg[:tokens_to_consider_in_iteration]]
		else:
			token_and_sg_log_prob_without_it = sg_bpe_model_object.get_sg_log_prob_without_tokens_mp2(current_total_sg, True, current_dict_of_top_tokens)
			sorted_tokens_sg = sorted(token_and_sg_log_prob_without_it.items(), key=lambda item: item[1])

		# finding the tokens to prune in that iteration (those that without them the value is minimal)
		tokens_to_prune = [t[0] for t in sorted_tokens_sg[:tokens_to_prune_in_iteration]]
		log2.info("pruning: {}".format(tokens_to_prune))
		#original_positions = [(t, list_of_sorted_tokens_sg.index(t)) for t in tokens_to_prune]
		#log2.info("their original positions in sg list were: {}".format(original_positions))
		
		for t in tokens_to_prune:
			try:
				current_vocab.remove(t)
				current_dict_of_top_tokens.remove(t)
			except:
				log2.info("could not remove {}".format(t[0]))
				raise

		# update our model vocab
		with open(vocab_filepath + ".bin", "wb") as vocab_file:
			pickle.dump(current_vocab, vocab_file)

		sg_bpe_model_object.set_vocab(current_vocab)
		sg_bpe_model_object.update_encoded_form_for_corpus_lines(tokens_to_prune)
		
		# upgrade our current total sg
		current_total_sg = sg_bpe_model_object.total_sg_log_prob(partial_corpus_filepath)

		# log current sg objective
		log2.info("current SG-BPE-500 total_log_sg_prob: {}".format(current_total_sg))
		log2.info("current size is {}".format(len(current_vocab)))
		log2.info("Time elapsed (iteration #{}) - {} (minutes)".format(iteration, float(progress_bar.format_dict["elapsed"])/60))
		log2.info("Time elapsed (iteration #{}) - {} (minutes)".format(iteration, float(progress_bar.format_dict["elapsed"])/60))

		# prepare to next iteration
		log2.log_separator()
		progress_bar.update(1)
		iteration += 1

	progress_bar.close()

	# Logging experiment results:
	sg_bpe_model_object.log_experiments_model_results(partial_corpus_filepath, "sg_bpe_500")

	final_total_skipgram_ns_probability = sg_bpe_model_object.total_sg_log_prob(partial_corpus_filepath)
	log2.info("Final SG-BPE-500 total_log_sg_prob: {}".format(final_total_skipgram_ns_probability))

	bpe_vanilla_total_skipgram_ns_probability = bpe_vanilla_model_object.total_sg_log_prob(partial_corpus_filepath)
	log2.info("Vanilla-BPE-500 total_log_sg_prob: {}".format(bpe_vanilla_total_skipgram_ns_probability))

	# Log difference between vanilla and updated bpe models
	# log tokens only in sg-bpe vocab
	current_vocab = sg_bpe_model_object.get_current_vocab()
	log2.info("vocab len = {}".format(len(current_vocab)))
	bpe_vanilla_500_vocab = bpe_vanilla_model_object.get_current_vocab()
	log2.info("bpe vanilla vocab len = {}".format(len(bpe_vanilla_500_vocab)))

	only_in_sg_bpe = [t for t in current_vocab if t not in bpe_vanilla_500_vocab]
	sg_bpe_500_only_filepath = "./results/{}/sg_bpe_500_only.txt".format(experiment_name)
	with open(sg_bpe_500_only_filepath, "w+") as sg_bpe_500_only:
		sg_bpe_500_only.write(json.dumps(only_in_sg_bpe, indent=4))

	# log tokens only in bpe-vanilla 500 vocab
	only_in_bpe_vanilla_500 = [t for t in bpe_vanilla_500_vocab if t not in current_vocab]
	bpe_vanilla_500_only_filepath = "./results/{}/bpe_vanilla_500_only.txt".format(experiment_name)
	with open(bpe_vanilla_500_only_filepath, "w+") as bpe_vanilla_500_only:
		bpe_vanilla_500_only.write(json.dumps(only_in_bpe_vanilla_500, indent=4))
		
	# Average token length
	average_token_length = sum(map(len, current_vocab)) / len(current_vocab)
	log2.info("SG-BPE average token length: {}".format(average_token_length))

	# Average token length of vanilla-bpe
	vanilla_average_token_length = sum(map(len, bpe_vanilla_500_vocab)) / len(bpe_vanilla_500_vocab)
	log2.info("Vanilla-BPE average token length: {}".format(vanilla_average_token_length))

	# Length of encoded file data
	with open(partial_corpus_filepath) as input_file:
		input_file_data = input_file.read()

	encoded_data = [sg_bpe_model.id_to_piece(x) for x in sg_bpe_model.encode(input_file_data)]
	log2.info("SG-BPE encoding length: {} (file length: {})".format(len(encoded_data), len(input_file_data)))

	# Length of encoded file data of vanilla-bpe
	vanilla_encoded_data = [bpe_vanilla_500_model.id_to_piece(x) for x in bpe_vanilla_500_model.encode(input_file_data)]
	log2.info("Vanilla-BPE encoding length: {} (file length: {})".format(len(vanilla_encoded_data), len(input_file_data)))

def prepare_parameters():
	parser = argparse.ArgumentParser(description="Calculating SaGe vocabulary.")
	parser.add_argument("experiment_name", help="name of experiment, will save results under that name.") # Required param (no "--" prefix)
	parser.add_argument("--final_vocab_size", required=True, help="final vocabulary size")
	parser.add_argument("--initial_vocab_size", required=True, help="initial vocabulary size")
	parser.add_argument("--tokens_to_prune_in_iteration", required=True, help="number of tokens to prune in each iteration")
	parser.add_argument("--tokens_to_consider_in_iteration", required=True, help="number of tokens to consider in each iteration")
	parser.add_argument("--iterations_until_reranking", required=True, help="number of iterations until reranking")
	parser.add_argument("--is_continue", default="N", help="is this execution continues former execiution of that experiment: [Y/N]")
	parser.add_argument("--corpus_filepath", required=True, help="filepath for full corpus (like wiki corpus)")
	parser.add_argument("--thousands_of_corpus_lines", default=200, help="number of corpus lines - in thousands")
	parser.add_argument("--partial_corpus_filepath", required=True, help="where to create partial corpus file - with number of lines requested")
	parser.add_argument("--max_lines_per_token", default=1000, help="max number of lines to consider in objective calculation, per-token")
	parser.add_argument("--window_size", default=5, help="window size for SG objective calculation, and also for wordpiece embeddings calculation")
	parser.add_argument("--use_gensim", default="N", help="Whether to use gensim to compute word embeddings [Y/N]")
	return vars(parser.parse_args())

if __name__ == "__main__":
	args = prepare_parameters()

	experiment_name = args["experiment_name"]
	print("Starting experiment {}".format(experiment_name))

	is_continue_execution = True if args["is_continue"] == "Y" else False
	print("is_continue={}".format(is_continue_execution))
	
	use_gensim = True if args["use_gensim"] == "Y" else False

	K_NUMBER = 1000
	partial_corpus_lines_number = int(args["thousands_of_corpus_lines"]) * K_NUMBER

	final_vocab_size = int(args["final_vocab_size"])
	initial_vocab_size = int(args["initial_vocab_size"])
	tokens_to_prune_in_iteration = int(args["tokens_to_prune_in_iteration"])
	tokens_to_consider_in_iteration = int(args["tokens_to_consider_in_iteration"])
	iterations_until_reranking = int(args["iterations_until_reranking"])
	max_lines_per_token = int(args["max_lines_per_token"])
	corpus_filepath = args["corpus_filepath"]
	partial_corpus_filepath = args["partial_corpus_filepath"]
	window_size = int(args["window_size"])

	main(experiment_name, is_continue_execution, final_vocab_size, \
		initial_vocab_size, partial_corpus_lines_number, tokens_to_prune_in_iteration, tokens_to_consider_in_iteration, \
		iterations_until_reranking, max_lines_per_token, corpus_filepath, partial_corpus_filepath, window_size, use_gensim)
