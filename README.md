# SaGe 2.0
Version 2.0 for the SaGe subword tokenizer ([EACL 2023](https://aclanthology.org/2023.eacl-main.45/)), excelling in [morphological segmentation](https://aclanthology.org/2024.acl-short.73/). Downstream applications of the tokenizer, i.e. pre-training an LLM model and evaluating on benchmarks, are independent of the tokenizer code - in the paper we used [academic budget BERT](https://github.com/IntelLabs/academic-budget-bert).

Pre-trained SaGe-based models are available in [this](https://github.com/kensho-technologies/timtc_vocabs_models) repository.
The large versions (2.4B params) produced the best results over BPE, UnigramLM, and PathPiece---see Table 14 in the Appendix [here](https://aclanthology.org/2024.emnlp-main.40/).

SaGe 2.0 implements a faster, parallelizable version of the vocabulary learning algorithm.

```python
from sage_tokenizer.SaGeVocabBuilder import SaGeVocabBuilder
vocab_builder = SaGeVocabBuilder(full_vocab_schedule=[262144, 229376, 196608, 163840, 131072, 98304, 65536, 57344, 49152, 40960, 32768, 16384], 
                                 embeddings_schedule=[262144, 131072, 65536, 49152, 40960, 32768],
                                 workers_number=4)

vocab_builder.build_vocab(experiment_name='experiment_name', 
                          corpus_filepath='data/wiki_lines.txt', 
                          vocabulary_filepath='data/initial_vocab_hex.vocab')                     
```
The `.vocab` file can then be loaded as-is into most tokenization toolkits, such as Huggingface's `tokenizers`.

SaGe tokenizer can be installed from PyPI:
```
pip install sage-tokenizer
```

## Requirements
1. `gensim==4.3.2`
2. `scipy==1.12.0`
3. a prepared corpus - see `Dataset` section.
4. an initial vocabulary - see `Dataset` section.

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

## Arguments

Required arguments:

- `experiment_name`: Positional first parameter - a unique name for the experiment. Results will be saved under that name (in the `results` directory).
- `corpus_filepath`: filepath for the full corpus (e.g. wiki corpus). Format is lines of raw text. A random subset from this file is used to create the partial corpus, which serves as the actual corpus for training.  
- `initial_vocabulary_filepath`: initial vocabulary, hex formatted, one vocab word per line. 
- `vocabulary_schedule`: what vocabulary sizes are we aiming for. **Note:** Tokenization won't be done for the last vocab size.
- `embeddings_schedule`: from vocabulary_schedule, in which steps we should re-run embeddings (similar to *l* in paper).
	
Default override arguments:

- `partial_corpus_filepath`: where to create / load partial corpus file. Default is `''` for creating partial corpus under 'data' folder. The partial corpus is a random subset of the full corpus and serves as the actual corpus used for training.
- `partial_corpus_line_number`: number of lines for partial corpus - in thousands. Default is `1000`.
- `max_len`: max length of tokens in bytes. Default is `16`.
- `workers`: number of worker threads to use. Default is `1`.
- `random_seed`: random seed value. Default is `692653`.

- **word2vec arguments:**
  - `word2vec_D`: word2vec embedding vector length. Default is `50`
  - `word2vec_N`: word2vec number of negative samples. Default is `15`
  - `word2vec_ALPHA`: word2vec Initial learning rate. Default is `0.025`
  - `word2vec_window_size`: word2vec context window size. Default is `5`
  - `word2vec_min_count`: word2vec minimum count of word. Default is `1`, i.e. must be used at least once
  - `word2vec_sg`: word2vec skip-gram if 1; otherwise CBOW. Default is `1`

For execution via command line, run `main.py` from its working directory.   

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
        --random_seed 692653
```

### API differences from **SaGe 1.0**:
- **Argument `final_vocab_size` is obsolete**, and replaced by the last value of `vocabulary_schedule`.  
- In **SaGe 1.0** the algorithm started with running *BPE* to create an initial vocab in a desired size (*V×n* in paper).  
Now, the algorithm accepts an already created vocabulary file as input, **making argument `initial_vocab_size` obsolete**. 
- In **SaGe 1.0**, after the initial vocabulary created, we iteratively ablated constant number of tokens (*k* in paper), 
until the final vocab size (*V*) was reached.   
Now, the user can directly choose the intermediate vocabulary sizes, thereby defining the ablation schedule (i.e., the difference between each adjacent vocab sizes), **making the ablation size dynamic and `tokens_to_prune_in_iteration` obsolete**. 
- Due to performance improvement, reranking happens every iteration and all tokens are considered for ablation, **making arguments `iterations_until_reranking` (*m* in paper) and `tokens_to_consider_in_iteration` (*M* in paper) obsolete**. 
- In order to re-execute from a checkpoint, just execute the same command. By default, the script searches for already existing files under `results/exp_name` directory. **Argument `is_continue` is obsolete**.
- Argument `thousands_of_corpus_lines` name changed to `partial_corpus_line_number`. 
- Argument `max_lines_per_token` is obsolete, all lines will be considered for each token in the objective calculation.

## Citation

Version 2.0 was mostly developed by Kensho Technologies, LLC, and ported by Bar Gazit. Citation TBA.

Version 1.0 was developed by Shaked Yehezkel and Yuval Pinter, please use this citation:
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
