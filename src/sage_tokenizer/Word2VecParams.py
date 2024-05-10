# Copyright © 2023 Kensho Technologies, LLC

class Word2VecParams:
    def __init__(self, D, N, ALPHA, window_size, min_count, sg):
        self.D = D
        self.N = N
        self.ALPHA = ALPHA
        self.window_size = window_size
        self.min_count = min_count
        self.sg = sg
