# SaGe
Code for the SaGe subword tokenizer ([EACL 2023](https://aclanthology.org/2023.eacl-main.45/)). Downstream applications of the tokenizer, i.e. pre-training an LLM model and evaluating on benchmarks, are independent of the tokenizer code - in the paper we used [academic budget BERT](https://github.com/IntelLabs/academic-budget-bert).


## Requirements
1. `sentencepiece` (version 0.1.95)
2. `gensim`
3. a prepared corpus - see `Dataset` section.

This code does not require GPU\TPU.

## Dataset
In the paper, we used wikipedia latest dumps - `wget https://dumps.wikimedia.org/<XX>wiki/latest/<XX>wiki-latest-pages-articles.xml.bz2`, where `<XX>` should be the language code (`en`, `tr`, etc.).

You can use any other dataset instead, the format for the script is one file with lines of raw text.

## Notes
This script can be re-executed "from checkpoint" -
The vocabulary creation script saves several files ("checkpoints") to be able to later continue - for example it saves the seed, the embeddings and sentencepiece models, and even a list of tokens sorted according to the Skipgram objective.

## Execution
Execute `Main.py` from its working directory.
The command line parameters are:
	1. `experiment_name`: positional first parameter. A unique name for the experiment, results will be saved under that name (in the `results` directory).

Required arguments:
```	
	--final_vocab_size: expected final vocabulary size.
	--initial_vocab_size: initial vocabulary size, from which ablation start.
	--tokens_to_prune_in_iteration: number of tokens to prune in each iteration ($k$ in paper).
	--tokens_to_consider_in_iteration: number of tokens to consider in each iteration ($M$ in paper).
	--iterations_until_reranking: number of iterations until reranking ($m$ in paper).
	--corpus_filepath: filepath for the full corpus (e.g. wiki corpus).
	--partial_corpus_filepath: where to create a partial corpus file in case of thousands_of_corpus_lines argument supplied.
```
	
Default override arguments:
```
	--is_continue: is this execution continuing a former execiution of the same experiment: [Y/N]. default="N".
	--thousands_of_corpus_lines: number of corpus lines - in thousands. default=200.
	--max_lines_per_token: max number of lines to consider for each token in the objective calculation (not mentioned in the paper, affects a small portion of the vocabulary responsible for many unnecessary calculations). default=1000.
	--window_size: window size for the Skipgram objective calculation, as well as for the embeddings calculation. default=5.
```

Example:
```    
python Main.py \
        exp_name \
        --final_vocab_size 16000 \
        --initial_vocab_size 20000 \
        --tokens_to_prune_in_iteration 50 \
        --tokens_to_consider_in_iteration 2000 \
        --iterations_until_reranking 15 \
        --corpus_filepath data/wiki_lines.txt \
        --partial_corpus_filepath data/wiki_lines_partial.txt \
        --thousands_of_corpus_lines 2000
```

To re-execute from a checkpoint, just execute the same command with `--is_continue Y`. The script will then search for those files under the `results/exp_name` directory.

## Citation
```
@inproceedings{yehezkel-pinter-2023-incorporating,
    title = "Incorporating Context into Subword Vocabularies",
    author = "Yehezkel, Shaked  and
      Pinter, Yuval",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.45",
    pages = "623--635",
}
```
