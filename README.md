# SaGe
Code for SaGe subword tokenizer (EACL 2023).




Requirements:
-------------
In order to create vocabulary, you first have to:

1. install sentencepiece (version 0.1.95).
2. have your corpus - see "Dataset" section later.
3. mkdir results (if not already exists) - the script will save state under "results/<experiment-name>".
4. may be some more packages.

This code does not require GPU\TPU.

Dataset:
--------
1. We used wikipedia latest dumps - wget https://dumps.wikimedia.org/<XX>wiki/latest/<XX>wiki-latest-pages-articles.xml.bz2, where <XX> should be the language code (en, tr, etc.).
2. Bzip2 -vdk <XX>wiki-latest-pages-articles.xml.bz2 (unzip)

In this step we get XML of wikipedia articles.

3. Preprocess: (This command may take some time, even couple of hours)
from the academic-budget-bert/dataset directory, execute:
python process_data.py -f <XX>wiki-latest-pages-articles.xml -o data/ --type wiki

In this step we obtain "wiki_one_article_per_line.txt", which is in the right format for the vocab creation script.
Anyway, it contains veryyyy long lines - every line is an article, which seems to be too much for the script… 

4. So you might want to execute "preprocess_for_sage.py" too. (from \utils in this repo).
Now you have file with short wiki lines.

** You could have used any other dataset instead, the format for the script is one file with lines of text **

Notes:
------
1. This script can be re-executed "from checkpoint" -
Slurm can cancel your job anytime, so you better have scripts that can be re-executed.
In our case, the vocabulary creation script saves several files ("checkpoints") to be able to later continue - for example it saves the seed, the embeddings and sentencepiece models, and even a list of tokens sorted by our objective.
If you pass "--is_continue Y" the script searches for those files under the "results/experiment_name" directory.

Execution:
----------
Execute Main.py from its working directory.
Before you do that, create a directory "results" in the same working directory.
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
	--window_size: window size for SG objective calculation, and also for wordpiece embeddings calculation. default=5.
	--use_gensim: Whether to use gensim to compute word embeddings [Y/N]. default="N".

Example:
    python Main.py \
        create_16k_vocab \
        --final_vocab_size 16000 \
        --initial_vocab_size 20000 \
        --tokens_to_prune_in_iteration 50 \
        --tokens_to_consider_in_iteration 2000 \
        --iterations_until_reranking 15 \
        --corpus_filepath "../data/wiki_lines.txt" \
        --partial_corpus_filepath "../data/wiki_lines_partial.txt" \
        --thousands_of_corpus_lines 500 \
        --use_gensim Y 

If this job would be canceled and we like to re-execute it from where it stopped, just execute the same command with "--is_continue Y".

Information about the code:
---------------------------
sg_bpe_model_object.get_sg_log_prob_without_tokens_mp2 is the body of the pruning loop.
It uses python multiprocessing because python GIL makes multithreading not helpful.
It makes python to call "utils.get_diff_sg_wo_token_for_line" for every relevant token, in its own process so they run in parallel.
