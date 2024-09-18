python main.py exp_name \
    --corpus_filepath data/wiki_lines.txt \
    --initial_vocabulary_filepath data/initial_vocab_hex.vocab \
    --vocabulary_schedule 262144 229376 196608 163840 131072 98304 65536 57344 49152 40960 32768 16384 \
    --embeddings_schedule 262144 131072 65536 49152 40960 32768 \
    --partial_corpus_filepath data/wiki_lines_partial.txt \
    --partial_corpus_line_number 500 \
    --max_len 17 \
    --workers 4 \
    --random_seed 692653
