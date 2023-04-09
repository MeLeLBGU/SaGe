# SaGe
Code for SaGe subword tokenizer (EACL 2023).




Requirements:
-------------
1. sentencepiece (version 0.1.95).
2. have your corpus - see "Dataset" section later.

This code does not require GPU\TPU.

Dataset:
--------
In the paper, we used wikipedia latest dumps - wget https://dumps.wikimedia.org/<XX>wiki/latest/<XX>wiki-latest-pages-articles.xml.bz2, where <XX> should be the language code (en, tr, etc.).

You can use any other dataset instead, the format for the script is one file with lines of text.

Notes:
------
This script can be re-executed "from checkpoint" -
The vocabulary creation script saves several files ("checkpoints") to be able to later continue - for example it saves the seed, the embeddings and sentencepiece models, and even a list of tokens sorted by our objective.
If you pass "--is_continue Y" the script searches for those files under the "results/experiment_name" directory.

Execution:
----------
Execute Main.py from its working directory.
The command line parameters are:
	1. experiment_name: positional first parameter. name of experiment, will save results under that name (in the "results" directory).

Required arguments:
	--final_vocab_size: final vocabulary size.
	--initial_vocab_size: initial vocabulary size.
	--tokens_to_prune_in_iteration: number of tokens to prune in each iteration.
	--tokens_to_consider_in_iteration: number of tokens to consider in each iteration.
	--iterations_until_reranking: number of iterations until reranking.
	--corpus_filepath: filepath for full corpus (like wiki corpus).
	--partial_corpus_filepath: where to create partial corpus file - with number of lines requested.
	
Default override arguments:
        --is_continue: is this execution continues former execiution of that experiment: [Y/N]. default="N".
	--thousands_of_corpus_lines: number of corpus lines - in thousands. default=200.
	--max_lines_per_token: max number of lines to consider in objective calculation, per-token. default=1000.
	--window_size: window size for SG objective calculation, and also for embeddings calculation. default=5.

Example:
    python Main.py \
        exp_name \
        --final_vocab_size 16000 \
        --initial_vocab_size 20000 \
        --tokens_to_prune_in_iteration 50 \
        --tokens_to_consider_in_iteration 2000 \
        --iterations_until_reranking 15 \
        --corpus_filepath "../data/wiki_lines.txt" \
        --partial_corpus_filepath "../data/wiki_lines_partial.txt" \
        --thousands_of_corpus_lines 2000

To re-execute it from where it stopped, just execute the same command with "--is_continue Y".
