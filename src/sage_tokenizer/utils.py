# Copyright © 2023 Kensho Technologies, LLC
from typing import Iterable, Tuple, List, Optional, Dict

import json
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from .model import SaGeTokenizer

# only log code outside of multiprocessing
# logger = logging.getLogger(__name__)
from .paths import getDataFolder, getLogsFolder, getResultsFolder


def write_vocab(vocab: Dict[bytes,int], filename: Path):
    """
    Dump the byte vocab to a file, encoded as hex characters inside this function.
    Saved in same order by index, so should preserve order.
    No special tokens are added.
    """
    # write these in increasing index order
    # so same as any previous order
    byindex = sorted([(idx, token) for token, idx in vocab.items()])

    with open(filename, 'w', encoding="utf-8") as f:
        for _, token in byindex:
            f.write(token.hex() + '\n')


def save_sorted_losses(sage_model: SaGeTokenizer, sorted_losses, target_vocab_size: int, vocab_folder: Path):
    vocab_folder = Path(vocab_folder)

    sorted_losses_filepath = vocab_folder / f"sorted_losses_before_{target_vocab_size}.txt"
    worst_500_filepath     = vocab_folder / f"worst_500_{target_vocab_size}.txt"
    best_500_filepath      = vocab_folder / f"best_500_{target_vocab_size}.txt"

    logging.info(f"Saving sorted losses to {sorted_losses_filepath.as_posix()}")
    write_sorted_losses_into_file(sorted_losses,   sorted_losses_filepath, sage_model)
    write_sorted_losses_into_file(sorted_losses[:500], worst_500_filepath, sage_model)
    write_sorted_losses_into_file(sorted_losses[-500:], best_500_filepath, sage_model)


def write_sorted_losses_into_file(sl: Iterable[Tuple[float,int]], filename: Path, sage_model: SaGeTokenizer):
    with open(filename, 'w', encoding="utf-8") as f:
        for loss, tid in sl:
            f.write(sage_model.id_to_encoded(tid) + "\t" + str(loss) + "\n")


def load_vocab(vocab_filepath: Path) -> List[bytes]:
    """
    Read our hex formatted vocab file, return a list of bytes objects.
    Input file has one vocab word per line, each hex encoded.
    """
    vocab_filepath = Path(vocab_filepath)
    if not vocab_filepath.exists():
        raise FileNotFoundError(f'Missing vocab file: {vocab_filepath.as_posix()}')

    with open(vocab_filepath, "r") as vocab_file:
        # fromhex ignores whitespace from \n at end
        initial_vocab = [bytes.fromhex(token) for token in vocab_file.readlines()]

    return initial_vocab


def load_corpus(corpus_filepath: Path, partial_corpus_filepath: Optional[Path], partial_corpus_line_number: int) -> List[str]:
    corpus_filepath         = Path(corpus_filepath)
    partial_corpus_filepath = Path(partial_corpus_filepath) if isinstance(partial_corpus_filepath, str) else partial_corpus_filepath

    if partial_corpus_filepath and partial_corpus_filepath.exists():  # Corpus already exists, directly loading
        logging.info(f"Found pre-existing partial corpus. Loading from {partial_corpus_filepath.as_posix()}...")
        read_start = time.time()
        with open(partial_corpus_filepath, "r") as corpus_f:
            partial_corpus = corpus_f.readlines()
        logging.info(f"Size of Corpus: {len(partial_corpus)}, time: {(time.time() - read_start):.2f}")
    else:
        read_start = time.time()
        with open(corpus_filepath, "r") as full_corpus_f:
            corpus = full_corpus_f.readlines()
            logging.info(f"Loading from Original Corpus. Number of lines: {len(corpus)}")

        random.shuffle(corpus)
        logging.info(f"Original Corpus read and shuffled. Time: {(time.time() - read_start):.2f}")

        # may be same as original depending on partial_corpus_line_number
        write_start_time = time.time()
        partial_corpus = corpus[:partial_corpus_line_number * 1000]

        if partial_corpus_filepath is None:
            partial_corpus_filepath = getDataFolder() / f"{corpus_filepath.stem}_{len(partial_corpus)}.txt"

        with open(partial_corpus_filepath, "w+") as partial_corpus_f:
            partial_corpus_f.writelines(partial_corpus)
        logging.info(f"Partial corpus saved at {partial_corpus_filepath.as_posix()}. "
                     f"Number of lines: {len(partial_corpus)}, "
                     f"time: {(time.time() - write_start_time):.2f}")

    return partial_corpus


def divide_data_by_num(data: List[str], num_procs: int) -> Iterable[List[str]]:
    """
    Split the data given the number of chunks we expect
    Returns a generator
    """
    size_per_chunk = len(data) // num_procs
    for i in range(0, len(data), size_per_chunk + 1):
        yield data[i: i + size_per_chunk + 1]


def divide_data_by_size(data, size):
    """
    Split the data given the size of chunks we expect
    Returns a generator
    """
    for i in range(0, len(data), size):
        yield data[i: i + size]


def compute_losses(losses, all_triples, embeddings):
    """
    function for computing losses given triple counts and embeddings
    losses : accumulate losses per ablated token, excluding the single byte ones, side effect this
    all_triples : triple values to aggregate into losses
    embeddings : embedding for each token
    """
    target_ids, context_ids, count = zip(*[(target_id, context_id, count) for (_, target_id, context_id), count in all_triples.items()])
    target_embeddings = np.array([embeddings[target_id] for target_id in target_ids])
    context_embeddings = np.array([embeddings[context_id] for context_id in context_ids])
    count = np.array(count)
    triples_loss = count * np.log(expit(np.einsum('ij,ij->i', target_embeddings, context_embeddings)))
    for idx, ((ablated_token_id, target_id, context_id), count) in enumerate(all_triples.items()):
        losses[ablated_token_id] = losses.get(ablated_token_id, 0.0) + triples_loss[idx]


def run_sage_parallel(embeddings: np.ndarray, partial_corpus: List[str], sage_model: SaGeTokenizer, workers_number: int):
    logging.info(f"Splitting data into {workers_number} chunks.")
    data_chunk_gen = divide_data_by_num(partial_corpus, workers_number)

    # these get aggregated over each chunk
    sage_losses = {}  # is token_id : loss
    overall_total_tokens = 0
    overall_total_triples = 0
    ablated_sizes = {}
    start_time = time.time()
    logging.info(f"Start spawning processes...")
    with mp.Pool(processes=workers_number) as pool:
        tasks = {}

        for tid, data_chunk in enumerate(data_chunk_gen):
            res = pool.apply_async(sage_per_chunk, args=(tid, sage_model, data_chunk, embeddings))
            tasks[res] = tid

        while tasks:
            results_ready_list = []
            for res, tid in tasks.items():
                if res.ready():
                    results_ready_list.append((res, tid))

            # process finished task results
            for res, tid in results_ready_list:
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

                logging.info(f"task {tid} finished after {(time.time() - start_time):.2f} seconds. "
                             f"Tokens:{total_tokens}, triples:{total_triples}, active:{len(sage_losses)}")

            logging.info(f"Sleeping 1 second. Number of still running tasks: {len(tasks)}")
            time.sleep(1.0)
    return overall_total_tokens, overall_total_triples, sage_losses, ablated_sizes


def sage_per_chunk(tid, model: SaGeTokenizer, data, embeddings, chunk_size: int=10000):
    """
    function that runs sage on each chunk of data (in parallelization)
    note: this is called from multiprocessing, so use print rather than logging
    """
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
        if (row > 0) and (row % chunk_size == 0):
            # take the total time here over all calls
            fs_time = time.time() - fs_start
            total_fs_time += fs_time
            # reinitialize fs_start
            fs_start = time.time()

            cl_start = time.time()
            compute_losses(losses, triples, embeddings)
            cl_time = time.time() - cl_start
            total_cl_time += cl_time

            print(f"fast_sage {tid}, row {row} of {len(data)}, "
                  f"fs_time: {fs_time:.2f}, cl_time: {cl_time:.2f}, "
                  f"triples: {len(triples)}, tokens: {total_tokens}")

            # total these up
            total_triples += len(triples)

            # zero out the triples from this chunksize lines
            triples = {}

    # compute for final partial chunk
    if triples:
        compute_losses(losses, triples, embeddings)
        total_triples += len(triples)

    # the triples can get quite large, so to avoid merging these
    # dict values, let's compute the losses in parallel too
    print(f"final fast_sage {tid}, row {row} of {len(data)}, "
          f"fs_time: {total_fs_time:.2f}, cl_time: {total_cl_time:.2f}, time: {(time.time() - start_time):.2f}, "
          f"triples: {len(triples)}, tokens: {total_tokens}")

    # Extra negative sign for equation (1) in SaGe paper
    # track number in cache too
    losses = {k: -v for k, v in losses.items()}

    return losses, total_tokens, total_triples, ablated_sizes


def init_logger(experiment_name: str):
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_filename = getLogsFolder() / f"{experiment_name}_{timestamp_str}.log"
    logging.basicConfig(filename=log_filename.as_posix(),
                        format="%(asctime)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    print(f"Logs will be stored in {log_filename.as_posix()}")


def get_output_folder(experiment_name: str) -> Tuple[Path, Path, Path]:
    results_path = getResultsFolder() / experiment_name

    vocab_folder = results_path / "sage_vocabs"
    vocab_folder.mkdir(exist_ok=True)

    stats_folder = results_path / "stats"
    stats_folder.mkdir(exist_ok=True)

    embeddings_folder = results_path / "embeddings"
    embeddings_folder.mkdir(exist_ok=True)

    return embeddings_folder, stats_folder, vocab_folder


def set_random_seed(experiment_name: str, random_seed: int):
    # Log seed
    seed_filepath = getResultsFolder() / experiment_name / "seed.txt"
    with open(seed_filepath, "w+") as f:
        f.write(str(random_seed))

    # Set seed
    random.seed(random_seed)
    np.random.seed(random_seed)


def save_stats(stats: dict, stats_folder: Path, target_vocab_size: int):
    stats_folder = Path(stats_folder)

    stats_filename = stats_folder / f"stats_{target_vocab_size}.json"
    logging.info(f"Saving stats to {stats_filename.as_posix()}")
    with open(stats_filename, "w") as f:
        json.dump(stats, f, indent=2)  # pretty print a bit
        f.write("\n")
