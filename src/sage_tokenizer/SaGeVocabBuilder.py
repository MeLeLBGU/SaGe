# Copyright © 2023 Kensho Technologies, LLC
from typing import List, Union, Optional
from pathlib import Path

import logging

from .Word2VecParams import Word2VecParams
from .embeddings import get_embeddings
from .model import SaGeTokenizer
from .utils import init_logger, set_random_seed, load_vocab, get_output_folder, load_corpus, run_sage_parallel, \
    save_sorted_losses, save_stats, write_vocab


class SaGeVocabBuilder:

    def __init__(self, full_vocab_schedule: List[int], embeddings_schedule: List[int],
                 max_len: int=16, workers_number: int=1, random_seed: int=692653,
                 word2vec_d: int=50, word2vec_n: int=15, word2vec_alpha: float=0.025, word2vec_window_size: int=5, word2vec_min_count: int=1, word2vec_sg: bool=True):
        self.full_vocab_schedule = full_vocab_schedule
        self.embeddings_schedule = embeddings_schedule
        self.max_len = max_len
        self.workers_number = workers_number
        self.random_seed = random_seed
        self.word2vec_params = Word2VecParams(
            D=word2vec_d,
            N=word2vec_n,
            ALPHA=word2vec_alpha,
            window_size=word2vec_window_size,
            min_count=word2vec_min_count,
            sg=int(word2vec_sg)  # 1 uses skip-gram, 0 uses CBoW.
        )

    def build_vocab(self, experiment_name: str,
                    corpus_filepath: Union[str,Path], vocabulary_filepath: Union[str,Path],
                    partial_corpus_filepath: Optional[Union[str,Path]]=None, partial_corpus_line_number: int=1000):
        corpus_filepath         = Path(corpus_filepath)
        vocabulary_filepath     = Path(vocabulary_filepath)
        partial_corpus_filepath = Path(partial_corpus_filepath) if isinstance(partial_corpus_filepath, str) else None

        init_logger(experiment_name)
        logging.info(f"Start experiment {experiment_name}")
        logging.info(f"Process will use up to {self.workers_number} worker threads.")

        logging.info("Getting output directories")
        embeddings_folder, stats_folder, vocab_folder = get_output_folder(experiment_name)
        logging.info("Setting random seed")
        set_random_seed(experiment_name, self.random_seed)
        logging.info(f"Loading initial vocabulary from file {vocabulary_filepath.as_posix()}")
        byte_vocab = load_vocab(vocabulary_filepath)
        logging.info(f"Finished loading initial vocabulary. Vocabulary size: {len(byte_vocab)}")

        actual_max_len = max([len(v) for v in byte_vocab])
        if self.max_len != actual_max_len:
            logging.warning(f"max_len parameter value {self.max_len} doesn't match actual max {actual_max_len}")

        logging.info("Initializing tokenizer")
        sage_model = SaGeTokenizer(byte_vocab, self.max_len)

        logging.info(f"Loading Corpus from {corpus_filepath.as_posix()}")
        partial_corpus = load_corpus(corpus_filepath, partial_corpus_filepath, partial_corpus_line_number)
        logging.info("Starting the training loop")
        vocab_schedule = self.full_vocab_schedule

        if not len(vocab_schedule) >= 2:
            raise Exception("Vocabulary schedule must contain more than 2 vocabulary sizes!")

        vocab_schedule.sort(reverse=True)  # largest first
        logging.info(f"initial vocab_schedule is {vocab_schedule[0]} vs actual size {sage_model.vocab_size()}")

        embedding_sizes = set(self.embeddings_schedule)

        # initialize embeddings for first iteration
        embeddings = get_embeddings(vocab_schedule[0], embeddings_folder, partial_corpus, sage_model,
                                        self.workers_number, self.word2vec_params)

        # skipping the initial vocab size here
        i = 0
        # stop one before the end, since we do i+1,
        # so we'll make the vocab of that final size, but won't do the tokenization on it
        while i < len(vocab_schedule) - 1:
            current_step_vocab_size = vocab_schedule[i]  # this will be the label used for files
            target_vocab_size = vocab_schedule[i + 1]
            actual_vocab_size = sage_model.vocab_size()
            logging.info(f"\nRound {i} - Start: "
                         f"\n\tCurrent step vocabulary size: {current_step_vocab_size}, "
                         f"\n\tTarget vocabulary size: {target_vocab_size}, "
                         f"\n\tActual vocabulary size: {actual_vocab_size}")

            if vocab_schedule[i] in embedding_sizes:
                embeddings = get_embeddings(current_step_vocab_size, embeddings_folder, partial_corpus, sage_model,
                                                self.workers_number, self.word2vec_params)

            if actual_vocab_size <= target_vocab_size:
                logging.info(f"Actual vocab is already smaller than target. continue to next iteration ")
                i += 1
                continue

            # call sage in parallel
            logging.info(f"Sage started.")
            total_tokens, total_triples, token_to_losses, ablated_sizes = run_sage_parallel(embeddings,
                                                                                            partial_corpus,
                                                                                            sage_model,
                                                                                            self.workers_number)
            logging.info(f"Sage finished. total tokens: {total_tokens}, total triplets: {total_triples}")

            # token_to_losses won't include any single byte tokens, but we want to keep those around
            # so lets just add them with large scores, so they stay around
            vocab_size_before_single_byte_tokens_addition = len(token_to_losses)
            sage_model.add_all_byte_ids(token_to_losses, score=1e6)
            logging.info(f"Adding single bytes to vocab. Size before: {vocab_size_before_single_byte_tokens_addition}, "
                         f"size after: {len(token_to_losses)}")

            # if a token doesn't appear in token_to_losses then it didn't participate in the tokenization
            current_active_vocab_size = len(token_to_losses)
            current_inactive_vocab_size = actual_vocab_size - len(token_to_losses)
            logging.info(f"Actual vocab size: {actual_vocab_size}, "
                         f"Target vocab size: {target_vocab_size}, "
                         f"Active Vocab Size: {current_active_vocab_size}, "
                         f"Inactive Vocab Size: {current_inactive_vocab_size}")

            # how many of the losses are negative
            neg_loss = len([loss for loss in token_to_losses.values() if loss < 0.0])
            zero_loss = len([loss for loss in token_to_losses.values() if loss == 0.0])
            pos_loss = len([loss for loss in token_to_losses.values() if loss > 0.0])
            logging.info(f"Negative losses: {neg_loss}, zero losses: {zero_loss}, positive losses: {pos_loss}")

            # in case the active vocab we found is actually smaller than the target vocab,
            # change the target to the next one, until it's smaller than the vocab we found,
            # so the ablation part will actually do something
            while current_active_vocab_size <= target_vocab_size:
                logging.info(f"Active vocab size is {current_active_vocab_size} - "
                             f"smaller than target {target_vocab_size}. Moving to next target_vocab_size"
                             f"\n\n(Round number increased to {i + 1})\n")
                i += 1
                target_vocab_size = vocab_schedule[i + 1]
                logging.info(f"New target_vocab_size: {target_vocab_size}")

            num_tokens_to_prune = current_active_vocab_size - target_vocab_size
            logging.info(f"Num tokens to prune {num_tokens_to_prune}")

            ######################
            # do the ablation
            ######################
            # we want to drop the smallest (negative) values
            # these are the ones with the largest decrease in likelihood from dropping the ablated token
            sorted_losses = list(sorted([(loss, tid) for (tid, loss) in token_to_losses.items()]))
            save_sorted_losses(sage_model, sorted_losses, target_vocab_size, vocab_folder)

            stats = {
                "current_step_vocab_size": current_step_vocab_size, "total_tokens": total_tokens,
                "total_triples": total_triples, "current_active_vocab_size": current_active_vocab_size,
                "current_inactive_vocab_size": current_inactive_vocab_size, "neg_loss": neg_loss,
                "zero_loss": zero_loss, "pos_loss": pos_loss, "target_vocab_size": target_vocab_size,
                "num_tokens_to_prune": num_tokens_to_prune, "ablated_sizes": ablated_sizes,
            }
            save_stats(stats, stats_folder, target_vocab_size)

            # these are the tokens to be removed
            tokens_to_prune = {sage_model.id_to_bytes(tid) for (loss, tid) in sorted_losses[:num_tokens_to_prune]}
            # double check there are no single bytes tokens to prune here
            single_byte_tokens_to_prune = [token for token in tokens_to_prune if len(token) == 1]
            assert len(single_byte_tokens_to_prune) == 0

            # our active vocabulary *after* pruning
            # is active if it has an entry in token_to_losses
            active_vocab = {tok: tid for tok, tid in sage_model.get_vocabulary().items()
                            if tid in token_to_losses and tok not in tokens_to_prune}

            # our overall vocabulary after pruning
            target_vocab = {tok: tid for tok, tid in sage_model.get_vocabulary().items()
                            if tok not in tokens_to_prune}

            # the deleted items
            deleted_vocab = {tok: tid for tok, tid in sage_model.get_vocabulary().items()
                             if tok in tokens_to_prune}

            vocab_save_name = vocab_folder / f"sage_vocab_{target_vocab_size}.vocab"
            logging.info(f"Saving intermediate vocab of size {len(target_vocab)} to {vocab_save_name.as_posix()}")
            write_vocab(target_vocab, vocab_save_name)

            active_save_name = vocab_folder / f"active_vocab_{target_vocab_size}.vocab"
            logging.info(f"Saving active vocab of size {len(active_vocab)} to {active_save_name.as_posix()}")
            write_vocab(active_vocab, active_save_name)

            # save the deleted ones too for analysis, with the original size
            deleted_save_name = vocab_folder / f"deleted_vocab_{target_vocab_size}.vocab"
            logging.info(f"Saving deleted vocab of size {len(deleted_vocab)} to {deleted_save_name.as_posix()}")
            write_vocab(deleted_vocab, deleted_save_name)

            # now update the internal state of sage_model to use the new smaller vocab
            # pass in list of bytes keys, which keep insertion order
            sage_model.set_vocabulary(list(target_vocab.keys()))

            logging.info(f"\nRound {i} - End: "
                         f"\n\tCurrent step vocabulary size: {current_step_vocab_size}, "
                         f"\n\tTarget vocabulary size: {target_vocab_size}, "
                         f"\n\tActual vocabulary size:{len(active_vocab)}")

            # advance to next smaller size
            i += 1
