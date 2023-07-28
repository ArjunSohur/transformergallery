# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import math
from gensim.models import Word2Vec
from nltk import word_tokenize
import torch


# ----------------------------------------------------------------------------------------------------------------------
# Class
# ----------------------------------------------------------------------------------------------------------------------
class Embedder:
    # Initialization
    def __init__(self, input_seq: str):
        # Setting the input sequence in case we need to refer back to it
        self.input_seq = input_seq

        # defining a dimension for Word2Vec
        self.word2vec_dim = 150

        # Splitting our data into words to be embedded
        self.words = word_tokenize(input_seq)

        # Creating an empty tensor to be filled later
        self.word_tensor = torch.empty([len(self.words), self.word2vec_dim])

    # The method that actually vectorizes words
    def embed(self):
        for i in range(len(self.words)):
            vector = Word2Vec(self.words[i], vector_size=self.word2vec_dim)
            self.word_tensor[i] = vector

    # Using positional encoding since transformers are positionally agnostic
    # See our tutorial on transformers to understand what this crazy formula is actually saying
    def positional_encode(self):
        for i in range(len(self.words)):
            for j in range(self.word2vec_dim):
                if j % 2 == 0:
                    self.word_tensor[i][j] += math.sin(i / (10000 ** (j / self.word2vec_dim)))
                else:
                    self.word_tensor[i][j] += math.cos(i / (10000 ** ((j - 1) / self.word2vec_dim)))
        