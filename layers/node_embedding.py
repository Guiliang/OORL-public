import torch
import h5py
import numpy as np
import torch.nn.functional as F


class H5EmbeddingManager(object):
    def __init__(self, h5_path):
        f = h5py.File(h5_path, 'r')
        self.W = np.array(f['embedding'])
        print("embedding data type=%s, shape=%s" % (type(self.W), self.W.shape))
        self.id2word = f['words_flatten'][0].decode("utf-8").split('\n')
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def word_embedding_initialize(self, words_list, dim_size=300, scale=0.1, oov_init='random'):
        shape = (len(words_list), dim_size)
        self.rng = np.random.RandomState(42)
        if 'zero' == oov_init:
            W2V = np.zeros(shape, dtype='float32')
        elif 'one' == oov_init:
            W2V = np.ones(shape, dtype='float32')
        else:
            W2V = self.rng.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V[0, :] = 0
        in_vocab = np.ones(shape[0], dtype=np.bool)
        word_ids = []
        for i, word in enumerate(words_list):
            if word in self.word2id:
                word_ids.append(self.word2id[word])
            else:
                in_vocab[i] = False
        W2V[in_vocab] = self.W[np.array(word_ids, dtype='int32')][:, :dim_size]
        return W2V


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, dropout_rate=0.0, trainable=True, id2word=None,
                 embedding_oov_init='random', load_pretrained=False, pretrained_embedding_path=None):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.id2word = id2word
        self.dropout_rate = dropout_rate
        self.load_pretrained = load_pretrained
        self.embedding_oov_init = embedding_oov_init
        self.pretrained_embedding_path = pretrained_embedding_path
        self.trainable = trainable
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        if self.embedding_layer.weight.is_cuda:
            init_embedding_matrix = init_embedding_matrix.cuda()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def embedding_init(self):
        # Embeddings
        if self.load_pretrained is False:
            word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
            word_embedding_init[0, :] = 0
        else:
            embedding_initr = H5EmbeddingManager(self.pretrained_embedding_path)
            word_embedding_init = embedding_initr.word_embedding_initialize(self.id2word,
                                                                            dim_size=self.embedding_size,
                                                                            oov_init=self.embedding_oov_init)
            del embedding_initr
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        if x.is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        embeddings = F.dropout(embeddings, p=self.dropout_rate, training=self.training)
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask
