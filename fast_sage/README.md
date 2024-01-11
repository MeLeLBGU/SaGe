# SaGe
Code for the SaGe subword tokenizer ([EACL 2023](https://aclanthology.org/2023.eacl-main.45/)). Downstream applications of the tokenizer, i.e. pre-training an LLM model and evaluating on benchmarks, are independent of the tokenizer code - in the paper we used [academic budget BERT](https://github.com/IntelLabs/academic-budget-bert).


## Requirements
1. `gensim`
2. a prepared corpus - see `Dataset` section.
3. an initial vocabulary - see `Dataset` section.

This code does not require GPU\TPU.

## Dataset
In the paper, we used wikipedia latest dumps - `wget https://dumps.wikimedia.org/<XX>wiki/latest/<XX>wiki-latest-pages-articles.xml.bz2`, where `<XX>` should be the language code (`en`, `tr`, etc.). 
We used them to create the corpus. From that corpus we use BPE tokenizer to create our initial vocabulary. 

You can use any other dataset instead.
The expected format for the corpus is one file with lines of raw text.
The expected format for the initial vocabulary is one vocab word per line, hex formatted.

## Notes
This script can be re-executed "from checkpoint" -
The vocabulary creation script saves several files ("checkpoints") to be able to later continue - for example it saves the partial corpus used, seed, the embeddings, and even a list of tokens sorted according to the Skipgram objective.

## Execution
Execute `Main.py` from its working directory.
The command line parameters are:
```
	`experiment_name`: 	
		positional first parameter. A unique name for the experiment. 
		results will be saved under that name (in the `results` directory).
```
Required arguments:
```	
	--corpus_filepath: filepath for the full corpus (e.g. wiki corpus). Foramt is lines of raw text.
	--initial_vocabulary_filepath: initial vocabulary, hex formatted, one vocab word per line. 
	--vocabulary_schedule: what vocabulary sizes are we aiming for. Tokenization won't be done on the last value.
	--embeddings_schedule: from vocabulary_schedule, in which steps we should re-run embeddings
```
	
Default override arguments:
```
	--partial_corpus_filepath: where to create / load partial corpus file. 
                Default is empty string for creating partial corpus under 'data' folder.
	--partial_corpus_line_number: number of lines for partial corpus - in thousands. Default is 1000.
	--max_len: max length of tokens in bytes. Default is 16.
	--workers: number of worker threads to use. Default is max(1, mp.cpu_count()-1).
	--random_seed: random seed value. Default is random.randint(1, 1000).
	
	word2vec arguments:
	--word2vec_D: word2vec embedding vector length. Default is 50
	--word2vec_N: word2vec number of negative samples. Default is 15
	--word2vec_ALPHA: word2vec Initial learning rate. Default is 0.025
	--word2vec_window_size: word2vec context window size. Default is 5
	--word2vec_min_count: word2vec minimum count of word. Default is 1, i.e. must be used at least once
	--word2vec_sg: word2vec skip-gram if 1; otherwise CBOW. Default is 1
	
```

Example:
```    
python main.py \
        exp_name \
        --corpus_filepath data/wiki_lines.txt \
        --initial_vocabulary_filepath data/initial_vocab_hex.vocab \
        --vocabulary_schedule 262144 229376 196608 163840 131072 98304 65536 57344 49152 40960 32768 16384 \
        --embeddings_schedule 262144 131072 65536 49152 40960 32768 \
        --partial_corpus_filepath data/wiki_lines_partial.txt \
        --partial_corpus_line_number 500 \
        --max_len 17 \
        --workers 4 \
        --random_seed 1234
```

To re-execute from a checkpoint, just execute the same command. By default, the script searches for already existing files under `results/exp_name` directory.

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
