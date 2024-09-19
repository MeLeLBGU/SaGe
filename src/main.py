# Copyright © 2023 Kensho Technologies, LLC

import argparse

from sage_tokenizer.SaGeVocabBuilder import SaGeVocabBuilder


def load_args():
    parser = argparse.ArgumentParser(description="Optimized implementation of SaGe method")
    parser.add_argument("experiment_name",
                        help="name of experiment, will save results under that name.")
    parser.add_argument("--corpus_filepath", required=True,
                        help="filepath for full corpus (e.g. wiki corpus)")
    parser.add_argument("--initial_vocabulary_filepath", required=True,
                        help="initial vocabulary, hex formatted, one vocab word per line")
    parser.add_argument("--vocabulary_schedule", nargs="+", type=int, required=True,
                        help="what vocabulary sizes are we aiming for. Tokenization won't be done on the last value")
    parser.add_argument("--embeddings_schedule", nargs="+", type=int, required=True,
                        help="from vocabulary_schedule, in which steps we should re-run embeddings")
    parser.add_argument("--partial_corpus_filepath", default="",
                        help="where to create / load partial corpus file. "
                             "Default is empty string for creating partial corpus under 'data' folder")
    parser.add_argument("--partial_corpus_line_number", type=int, default=1000,
                        help="number of lines for partial corpus - in thousands. Default is 1000")
    parser.add_argument("--max_len", type=int, default=16,
                        help="max length of tokens in bytes. Default is 16")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of worker threads to use. Default is 1")
    parser.add_argument("--random_seed", type=int, default=692653,
                        help="random seed value. Default is 692653")

    # word2vec params
    parser.add_argument("--word2vec_D", type=int, default=50,
                        help="word2vec embedding vector length. Default is 50")
    parser.add_argument("--word2vec_N", type=int, default=15,
                        help="word2vec number of negative samples. Default is 15")
    parser.add_argument("--word2vec_ALPHA", type=float, default=0.025,
                        help="word2vec Initial learning rate. Default is 0.025")
    parser.add_argument("--word2vec_window_size", type=int, default=5,
                        help="word2vec context window size. Default is 5")
    parser.add_argument("--word2vec_min_count", type=int, default=1,
                        help="word2vec minimum count of word. Default is 1, i.e. must be used at least once")
    parser.add_argument("--word2vec_sg", type=int, default=1,
                        help="word2vec skip-gram if 1; otherwise CBOW. Default is 1")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = load_args()
    vocab_builder = SaGeVocabBuilder(
        full_vocab_schedule=args['vocabulary_schedule'],
        embeddings_schedule=args['embeddings_schedule'],
        max_len=args['max_len'],
        workers_number=args['workers'],
        random_seed=args['random_seed'],
        word2vec_d=args['word2vec_D'],
        word2vec_n=args['word2vec_N'],
        word2vec_alpha=args['word2vec_ALPHA'],
        word2vec_window_size=args['word2vec_window_size'],
        word2vec_min_count=args['word2vec_min_count'],
        word2vec_sg=args['word2vec_sg']
    )

    vocab_builder.build_vocab(
        experiment_name=args['experiment_name'],
        corpus=args['corpus_filepath'],
        initial_vocabulary=args['initial_vocabulary_filepath'],
        k_corpus_examples=args['partial_corpus_line_number'],

        corpus_cache=args['partial_corpus_filepath'] if args['partial_corpus_filepath'] else "corpus"
    )
