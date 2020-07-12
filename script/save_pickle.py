import numpy as np
import pandas as pd
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    './fast-text/wiki-news-300d-1M.vec', datatype=np.float32)

pd.to_pickle(model, "./fast-text/fasttext_gensim_model.pkl")
