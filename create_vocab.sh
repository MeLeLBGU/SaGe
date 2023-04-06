#!/bin/bash

# this is run with "sbatch create_vocab.sh"

#SBATCH --job-name=exp
#SBATCH --output=exp.out
#SBATCH --error=exp.err
#SBATCH --partition=cpu-killable

python Main.py \
    exp \
    --final_vocab_size 10 \
    --initial_vocab_size 20 \
    --tokens_to_prune_in_iteration 5 \
    --tokens_to_consider_in_iteration 10 \
    --iterations_until_reranking 1 \
    --corpus_filepath "../data/wiki_lines.txt" \
    --partial_corpus_filepath "../data/wiki_lines_partial.txt" \
    --thousands_of_corpus_lines 1
