# Copyright © 2023 Kensho Technologies, LLC

class ParamConfig:

    # word2vec params
    # D = Embedding vector length
    D = 50

    # N = Number of negative samples
    N = 15

    # ALPHA = Initial learning rate
    ALPHA = 0.025

    # context window size
    # i.e. 2 to the left, 2 to the right, and one in the middle
    window_size = 5

    # minimum count of word, i.e. must be used somewhere
    min_count = 1

    # skip-gram if 1; otherwise CBOW
    sg = 1

    # what sizes are we aiming for
    # go one beyond, so we save out the 16384 vocab too
    full_vocab_schedule = [262144, 229376, 196608, 163840, 131072, 114688, \
                           98304, 81920, 65536, 57344, 49152, 40960, 32768, 16384, 8192]

    # and which of these should we re-embed
    embeddings_schedule = [262144, 131072, 65536, 49152, 40960, 32768]

    # random seed for calculations
    random_seed = 1234


config = ParamConfig()