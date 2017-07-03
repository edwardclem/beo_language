#mapping sequences of natural language text to BEO embedding vectors

import numpy as np
import tensorflow as tf
from os import listdir
import scipy.io as sio

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

class BEORNN():
    def __init__(self, train_data, vector_directory, saver_loc = None, load_loc = None, test_data = None,
                 embed_size = 30, hidden_state_size=50, fc_size=50,
                 epochs=10, batch_size=10):
        '''
        :param train_data: training data for model with BEO object names + language.
        :param vector_directory: location of .mat files with BEO embedding vectors.
        :param test_data: test data.
        :param embed_size: word embedding size.
        :param hidden_state_size: size of RNN hidden state.
        :param fc_size: fully connected output layer size.
        :param epochs: number of training epochs
        :param batch_size: training batch size
        '''

        #initialize network params
        self.embed_size, self.hidden_state_size, self.fc_size = embed_size, hidden_state_size, fc_size,
        #initialize training params
        self.epochs, self.batch_size = epochs, batch_size
        #saver params
        if saver_loc:
            self.saver_loc = saver_loc
        if load_loc:
            self.load_loc = load_loc

        #load data
        self.vector_dict, self.vec_size = self.__load_vectors__(vector_directory)
        self.training_data = self.__load_data__(train_data)
        self.train_lengths = [len(sent) for sent, vec in self.training_data]
        self.max_length = max(self.train_lengths)

        #tokenize data
        self.id2word, self.word2id = self.__build_vocab__()
        self.train_x, self.train_y = self.__vectorize__(self.training_data)

        #create initializer and session
        self.init = tf.truncated_normal_initializer(stddev=0.1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        #create placeholder for inputs and outputs
        self.X = tf.placeholder(tf.int32, shape=[None, self.max_length], name='desc_input')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.vec_size], name="BEO_vec")
        self.X_len = tf.placeholder(tf.int32, shape=[None], name="desc_len")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")

        #create inference graph
        self.predicted_vec = self.__inference__()

        #loss computation
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.predicted_vec, self.Y))

        #training operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        #saver
        self.saver = tf.train.Saver()

        #initialize variables
        self.session.run(tf.global_variables_initializer())

        #load if load location provided
        if self.load_loc:
            self.saver.restore(self.session, load_loc)


    @staticmethod
    def __load_vectors__(vec_dir):
        '''
        :param vec_dir: directory containing trained BEO vectors
        :return: dictionary of vector names, shape of vector
        '''

        vec_dict = {}
        for f in listdir(vec_dir):
            m = sio.loadmat("{}/{}".format(vec_dir, f))
            vec_dict[f.replace("_vec.mat", "")] = m['projectedObVec'][0]
        vec_size = vec_dict[vec_dict.keys()[0]].shape
        return vec_dict, vec_size[0]

    def __load_data__(self, data_loc):
        '''
        :param data_loc: location of text file with data.
        :return: list of (sentence, np vector) pairs.
        '''

        with open(data_loc, 'r') as f:
            all_data = f.read()

        pairs = [elem.split("\n") for elem in all_data.split("\n\n")]

        #reversing order and mapping to vectors
        data = [(elem[1].split(), self.vector_dict[elem[0]]) for elem in pairs]

        return data

    def __build_vocab__(self):
        '''
        :return:
        '''

        vocab = set()
        for nl, _ in self.training_data:
            vocab.update(nl)

        id2word = [PAD, UNK] + list(vocab)
        word2id = {word: word_id for word_id, word in enumerate(id2word)}
        return id2word, word2id

    def __vectorize__(self, data):
        '''
        :param data: input data
        :return: vectorized data.
        '''

        nl_all, vecs = zip(*data)
        nl_vectorized = []
        for nl in nl_all:
            vectorized = np.zeros((self.max_length,), dtype=np.int32)
            for i, word in enumerate(nl):
                vectorized[i] = self.word2id.get(word, UNK_ID)
            nl_vectorized.append(vectorized)

        return nl_vectorized, vecs

    def __vectorize_sentence__(self, sentence):
        """
        converts natural language sentence to vectors.
        :param sentence:
        :return:
        """

        nl = sentence.split()
        length = len(nl)
        vectorized = np.zeros((self.max_length,), dtype=np.int32)
        for i in range(min(self.max_length, length)):
            vectorized[i] = self.word2id.get(nl[i], UNK_ID)

        return vectorized, length

    @staticmethod
    def __relu__(inp, shape, initializer, keep_prob):
        '''
        rectified linear unit layer.
        :param shape:
        :param initializer:
        :param keep_prob:
        :return:
        '''
        weights = tf.get_variable("weights", shape=shape, dtype=tf.float32, initializer=initializer)
        biases = tf.get_variable("biases", shape=[shape[-1]], dtype=tf.float32, initializer=initializer)
        return tf.nn.dropout(tf.nn.relu(tf.matmul(inp, weights) + biases), keep_prob)

    @staticmethod
    def __output__(inp, shape, initializer):
        weights = tf.get_variable("weights", shape=shape, dtype=tf.float32, initializer=initializer)
        biases = tf.get_variable("biases", shape=[shape[-1]], dtype=tf.float32, initializer=initializer)
        return tf.matmul(inp, weights) + biases

    def __inference__(self):
        '''
        Create inference graph.
        :return:
        '''

        #create embedding layer
        embed_var = tf.get_variable("embeddings", shape=[len(self.word2id), self.embed_size], dtype=tf.float32, initializer=self.init)
        embedding_layer = tf.nn.dropout(tf.nn.embedding_lookup(embed_var, self.X), self.keep_prob) #embedding layer with dropout

        #shape = [None, self.max_length, embed_size

        #create GRU cell
        gru_cell = tf.contrib.rnn.GRUCell(self.hidden_state_size)
        _, hidden_state = tf.nn.dynamic_rnn(gru_cell, embedding_layer, sequence_length=self.X_len, dtype=tf.float32)

        #first FC layer with ReLU activation
        with tf.variable_scope("fc1"):
            fc1 = self.__relu__(hidden_state, [self.hidden_state_size, self.fc_size], self.init, self.keep_prob)

        #second FC layer -- how necessary is this?
        with tf.variable_scope("fc2"):
            fc2 = self.__relu__(fc1, [self.fc_size, self.fc_size], self.init, self.keep_prob)

        #output layer
        with tf.variable_scope("output"):
            output = self.__output__(fc2, [self.fc_size, self.vec_size], self.init)

        return output

    def train(self):
        '''
        Trains model w.r.t. provided data. Using sidd's chunking code.
        :return:
        '''
        chunk_size = len(self.train_x)
        for e in range(self.epochs):
            curr_loss, batches = 0.0, 0.0
            for start, end in zip(range(0, len(self.train_x[:chunk_size]) - self.batch_size, self.batch_size),
                                  range(self.batch_size, len(self.train_x[:chunk_size]), self.batch_size)):
                loss, _ = self.session.run([self.loss, self.train_op],
                                           feed_dict={self.X: self.train_x[start:end],
                                                      self.X_len: self.train_lengths[start:end],
                                                      self.keep_prob: 0.5,
                                                      self.Y: self.train_y[start:end]})
                curr_loss += loss
                batches += 1
            print 'Epoch %s Average Loss:' % str(e), curr_loss / batches

        if self.saver_loc:
            print "saving trained model"
            self.saver.save(self.session, self.saver_loc)

    def phrase_to_vec(self, sentence):
        '''
        maps a natural language description to vector
        :param sentence:
        :return:
        '''

        vectorized, length = self.__vectorize_sentence__(sentence)
        predicted = self.session.run(self.predicted_vec, feed_dict={self.X: [vectorized],
                                                                    self.X_len: [length],
                                                                    self.keep_prob: 1.0})
        return predicted



