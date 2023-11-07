# Copyright © 2023 Kensho Technologies, LLC

import sys
import os
import time
import argparse, logging
import random
import json

import numpy as np
import multiprocessing as mp

sys.path.insert(0, "./fast_sage")
import utils
import model
import embeddings as emb
from params_config import config

def log_config(experiment_name):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("logs"):
        os.mkdir("logs")
    log_filename = f'logs/{experiment_name}_{timestr}.log'
    logging.basicConfig(filename=log_filename,
                        format="%(asctime)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    print(f"Logs will be stored in {log_filename}")

    return logging

def load_args():
    parser = argparse.ArgumentParser(description="Optimized implementation of SaGe method")
    parser.add_argument("experiment_name", help="name of experiment, will save results under that name.") # Required param (no "--" prefix)
    parser.add_argument("--corpus_filepath", required=True, help="filepath for full corpus (like wiki corpus)")
    parser.add_argument("--vocab_filepath", required=True, help="vocabulary file to initialize the model")
    parser.add_argument("--partial_corpus_filepath", default="", help="where to load partial corpus file, or empty string if none")
    parser.add_argument("--partial_corpus_lines_number", default=1000, help="number of partial corpus lines - in thousands")
    parser.add_argument("--max_len", default=16, help="max length of tokens in bytes")
    return vars(parser.parse_args())

def main(args):

    args = load_args()
    experiment_name = args['experiment_name']

    logger = log_config(experiment_name)

    # logging = log_config(experiment_name)
    logger.info(f"Start experiment {experiment_name}")

    logger.info("-----------------------------------------")
    logger.info(f"Setting up directories...")

    if not os.path.exists("results"):
        logger.info("'results/' not found. Creating one for storing results...")
        os.mkdir("results")

    results_path = f"results/{experiment_name}"
    if not os.path.exists(results_path):
        logger.info(f"Creating directory for results: {results_path}")
        os.mkdir(results_path)

    logger.info("Preparing directories for storing intermediate results")
    vocab_save_filepath = results_path + "/sage_vocabs"
    stats_save_filepath = results_path + "/stats"
    embeddings_save_filepath = results_path + "/embeddings"

    if not os.path.exists(vocab_save_filepath):
        os.mkdir(vocab_save_filepath)
    if not os.path.exists(stats_save_filepath):
        os.mkdir(stats_save_filepath)
    if not os.path.exists(embeddings_save_filepath):
        os.mkdir(embeddings_save_filepath)


    logger.info("-----------------------------------------")
    logger.info(f"Fixing random seed to {config.random_seed}")
    utils.fix_random_seed(experiment_name, config.random_seed)


    logger.info("-----------------------------------------")
    logger.info("Loading initial vocab...")
    vocab_filepath = args['vocab_filepath'] #
    byte_vocab = utils.load_vocab(vocab_filepath)
    logger.info(f"Loading vocabulary of size {len(byte_vocab)} from file {vocab_filepath}")

    max_len = int(args['max_len'])  # TODO: is cast needed?
    actual_max_len = max([len(v) for v in byte_vocab])
    if max_len != actual_max_len:
        logger.warning(f"max_len parameter value {max_len} doesn't match actual max {actual_max_len}")
        print(f"max_len parameter value {max_len} doesn't match actual max {actual_max_len}")

    # set up our tokenizer
    sage_model = model.SaGeTokenizer(byte_vocab, max_len)


    logger.info("-----------------------------------------")
    partial_corpus_filepath = args['partial_corpus_filepath']
    corpus_filepath = args['corpus_filepath']
    logger.info(f"Loading Corpus from {corpus_filepath}")
    partial_corpus_lines_number = int(args['partial_corpus_lines_number']) # TODO: cast needed?

    # load the data
    partial_corpus, partial_corpus_filepath = utils.load_corpus(sage_model, corpus_filepath, partial_corpus_filepath, partial_corpus_lines_number)

    logger.info("-----------------------------------------")
    logger.info("Starting the training loop")
    vocab_schedule = config.full_vocab_schedule
    # need to have at least two levels
    assert len(vocab_schedule) >= 2, "Insufficient vocab sizes for running the algorithm, need at least 2!"

    vocab_schedule.sort(reverse=True)  # largest first
    logger.info(f"initial vocab_schedule is {vocab_schedule[0]} vs actual size {sage_model.vocab_size()}")

    num_procs = mp.cpu_count()
    logger.info(f"{num_procs} number of processes available")

    # do we need to embed this before this size?
    embedding_sizes = set(config.embeddings_schedule)

    # skipping the initial vocab size here
    i = 0
    # stop one before the end, since we do i+1
    # so we'll make the vocab of that final size, but won't do the tokenization on it
    while i < len(vocab_schedule)-1:

        current_vocab_size = vocab_schedule[i]  # this will be the label used for files
        target_vocab_size = vocab_schedule[i+1]

        current_total_vocab_size = sage_model.vocab_size()
        logger.info(f"Round {i} - Starting, current_vocab_size:{current_total_vocab_size}, target_vocab_size:{target_vocab_size}, current_total_vocab_size:{current_total_vocab_size}")

        if vocab_schedule[i] in embedding_sizes:
            logger.info(f"Round {i} - Retraining Embeddings at vocab size {current_vocab_size}")

            # is there an embedding of this size
            embeddings_filepath = f"{embeddings_save_filepath}/embeddings_{current_vocab_size}.npy"

            if os.path.exists(embeddings_filepath):
                logger.info(f"Found trained embeddings. Directly loading from {embeddings_filepath}...")
                # context and target embeddings are the same so just keep one copy around
                embeddings = np.load(embeddings_filepath)
            else:
                logger.info(f"Start training embeddings with Word2Vec...")
                start_time = time.time()
                embeddings = emb.train_embeddings(sage_model, partial_corpus, num_procs-1)
                logger.info(f"Embeddings time:{time.time()-start_time}")
                logger.info(f"Save embeddings to {embeddings_filepath}")
                np.save(embeddings_filepath, embeddings, allow_pickle=True)

        logger.info(f"Round {i} - Current total vocab size: {current_total_vocab_size}. Target vocab size: {target_vocab_size}")
        if current_total_vocab_size <= target_vocab_size:
            logger.info(f"Round {i} - Already have a smaller vocab than the target")
            i += 1 # skip to next
            break

        start_time = time.time()

        logger.info(f"Round {i} - Splitting Data into chunks...")
        data_chunk_gen = utils.divide_data_by_num(partial_corpus, num_procs-1) # TODO: do we need one for a coordinating node?

        # which losses did we have this iteration
        # these get aggregated over each chunk
        sage_losses = {}  # is token_id : loss
        overall_total_tokens = 0
        overall_total_triples = 0
        ablated_sizes = {}

        logger.info(f"Round {i} - Start spawning processes...")
        with mp.Pool(processes=max(1,num_procs-1)) as pool:
            tasks = {}

            for tid, data_chunk in enumerate(data_chunk_gen):
                res = pool.apply_async(utils.sage_per_chunk, args=(tid, sage_model, data_chunk, embeddings))
                tasks[res] = tid

            while tasks:
                results_ready_list = []
                for res, tid in tasks.items():
                    if res.ready():
                        # logger.info(f"Task {tid} results are ready. Adding to list...")
                        results_ready_list.append((res,tid))

                if len(results_ready_list) > 0:
                    for res, tid in results_ready_list:

                        # get our completed results
                        losses, total_tokens, total_triples, ab_sizes = res.get()

                        # just add these to totals/maxes
                        overall_total_tokens += total_tokens
                        overall_total_triples += total_triples

                        # add to the overall tallys
                        for k, v in losses.items():
                            sage_losses[k] = sage_losses.get(k, 0) + v

                        # how many tokens needed to be examined
                        for k, v in ab_sizes.items():
                            ablated_sizes[k] = ablated_sizes.get(k, 0) + v

                        # all done with this,
                        # can delete from tasks without messing up iteration over list
                        del tasks[res]

                        logger.info(f"Round {i} - task:{tid}, tokens:{total_tokens}, triples:{total_triples}, active:{len(sage_losses)}, "
                            + f"remaining:{len(tasks)}, elapsed: {time.time()-start_time}")
                else:
                    # logger.info(f"No results are ready for aggregation yet... sleeping...{len(tasks)}, {time.time()-start_time}")
                    time.sleep(1.0)

        logger.info(f"Round {i} - Done {time.time()-start_time}, {overall_total_tokens}, {overall_total_triples}")

        # sage_losses won't include any single byte tokens,
        # but we want to keep those around
        # so lets just add them with large scores, so they stay around
        # TODO: large or small here?
        before_size = len(sage_losses)
        # make sure all the single byte token ids are in there
        sage_model.add_all_byte_ids(sage_losses, score=1e6)
        logger.info(f"Round {i} - Adding single bytes {before_size}, {len(sage_losses)}")

        current_active_vocab_size = len(sage_losses)
        current_inactive_vocab_size = current_total_vocab_size - len(sage_losses)

        # how many of the losses are negative
        neg_loss = len([l for l in sage_losses.values() if l < 0.0])
        zero_loss = len([l for l in sage_losses.values() if l == 0.0])
        pos_loss = len([l for l in sage_losses.values() if l > 0.0])
        logger.info(f"Round {i} - Negative loss: {neg_loss}, zero loss: {zero_loss}, positive loss: {pos_loss}")

        # if a token doesn't appear in sage_losses then it didn't participate in the tokenization
        logger.info(f"Round {i} - Current total vocab size: {current_total_vocab_size}. Target vocab size: {target_vocab_size}")
        logger.info(f"Round {i} - Active Vocab Size: {current_active_vocab_size}. Inactive Vocab Size: {current_inactive_vocab_size}")

        # find the next target_vocab_size smaller than what we have, so we have something to drop
        while current_active_vocab_size <= target_vocab_size:
            logger.info(f"Round {i} - Found active vocab ({current_active_vocab_size}) smaller than target ({target_vocab_size}). Moving to next target_vocab_size")
            i += 1
            target_vocab_size = vocab_schedule[i]
            logger.info(f"Round {i} - New target_vocab_size: {target_vocab_size}")

        num_tokens_to_prune = current_active_vocab_size - target_vocab_size
        logger.info(f"Round {i} - Num tokens to prune {num_tokens_to_prune}")

        ######################
        # do the ablation
        # TODO: double check we don't we want the largest losses to go?

        # I think the losses are negative, so we'll want to drop the smallest (negative) values
        # these are the ones with the largest decrease in likelihood from dropping the ablated token
        sorted_losses = list(sorted([(loss,tid) for (tid,loss) in sage_losses.items()]))
        sl_save_name = f"{vocab_save_filepath}/sorted_losses_before_{target_vocab_size}.txt"
        logger.info(f"Round {i} - Saving sorted losses to {sl_save_name}")
        utils.write_sorted_losses(sorted_losses, sl_save_name, sage_model)

        worst_500_save_name = f"{vocab_save_filepath}/worst_500_{target_vocab_size}.txt"
        utils.write_sorted_losses(sorted_losses[:500], worst_500_save_name, sage_model)

        best_500_save_name = f"{vocab_save_filepath}/best_500_{target_vocab_size}.txt"
        utils.write_sorted_losses(sorted_losses[-500:], best_500_save_name, sage_model)

        # what should we track per iteration
        stats = {
            "current_vocab_size" : current_vocab_size,
            "overall_total_tokens" : overall_total_tokens,
            "overall_total_triples" : overall_total_triples,
            "current_active_vocab_size" : current_active_vocab_size,
            "current_inactive_vocab_size" : current_inactive_vocab_size,
            "neg_loss" : neg_loss,
            "zero_loss" : zero_loss,
            "pos_loss" : pos_loss,
            "target_vocab_size" : target_vocab_size,
            "num_tokens_to_prune" : num_tokens_to_prune,
            "ablated_sizes" : ablated_sizes,
        }
        json_stats = json.dumps(stats, indent=2) # pretty print a bit
        stats_save_name = f"{stats_save_filepath}/stats_{target_vocab_size}.json"
        logger.info(f"Round {i} - Saving stats to {stats_save_name}")
        with open(stats_save_name, 'wt') as f:
            f.write(json_stats + "\n")

        # these are the tokens to be removed
        tokens_to_prune = set([sage_model.id_to_bytes(tid) for (loss,tid) in sorted_losses[:num_tokens_to_prune]])
        # double check there are no single bytes here
        for tok in tokens_to_prune:
            assert len(tok) > 1

        # our active vocabulary *after* pruning
        # is active if has an entry in sage_losses
        active_vocab = { tok : tid for (tok,tid) in sage_model.get_vocabulary().items() \
            if tid in sage_losses and tok not in tokens_to_prune}

        # our overall vocabulary after pruning
        target_vocab = { tok : tid for (tok,tid) in sage_model.get_vocabulary().items() \
            if tok not in tokens_to_prune}

        # the deleted items
        deleted_vocab = { tok : tid for (tok,tid) in sage_model.get_vocabulary().items() \
            if tok in tokens_to_prune}

        vocab_save_name = f"{vocab_save_filepath}/sage_vocab_{target_vocab_size}.vocab"
        logger.info(f"Round {i} - Saving intermediate vocab of size {len(target_vocab)} to {vocab_save_name}")
        utils.write_vocab(target_vocab, vocab_save_name)

        active_save_name = f"{vocab_save_filepath}/active_vocab_{target_vocab_size}.vocab"
        logger.info(f"Round {i} - Saving active vocab of size {len(active_vocab)} to {active_save_name}")
        utils.write_vocab(active_vocab, active_save_name)

        # save the deleted ones too for analysis, with the original size
        deleted_save_name = f"{vocab_save_filepath}/deleted_vocab_{target_vocab_size}.vocab"
        logger.info(f"Round {i} - Saving deleted vocab of size {len(deleted_vocab)} to {deleted_save_name}")
        utils.write_vocab(deleted_vocab, deleted_save_name)

        # now update the internal state of sage_model to use the new smaller vocab
        # pass in list of bytes keys, which keep insertion order
        sage_model.set_vocabulary(target_vocab.keys())

        logger.info(f"Round {i} - End: {current_vocab_size}, target: {target_vocab_size}, final:{len(active_vocab)}, time:{time.time() - start_time}")

        # advance to next smaller size
        i += 1

if __name__ == '__main__':
    args = load_args()
    main(args)