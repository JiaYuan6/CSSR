import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import accuracy_score
import numpy as np
from utils import *


class Model:
    """
    GNN
    """

    def __init__(self, dimension, f_dim, f_out_size, item_n, batch_size=100, hide_size=100, out_size=100,
                 step=3,
                 alpha=0.1, beta=0.1):
        # self.vector = tf.placeholder(dtype=float, shape=[None, None])
        # 代码库的嵌入
        self.embeddings = tf.placeholder(dtype=tf.float32, name='embeddings', shape=[None, dimension])
        self.other_feat = tf.placeholder(dtype=tf.float32, name='feat1', shape=[None, f_dim])
        # session序列
        self.item = tf.placeholder(dtype=tf.int32, name='items')
        #self.mask = tf.placeholder(dtype=tf.int32, name='mask')
        self.n_session = tf.placeholder(dtype=tf.int32, name='session_num')
        self.n_node = tf.shape(self.embeddings)[0]
        self.item_num = item_n
        # 嵌入的向量
        self.dimension = dimension
        # 其他特征的嵌入维度
        self.f_dim = f_dim
        self.f_out_size = f_out_size
        # 输出向量的维度
        self.out_size = out_size
        # gru单元存在几层
        self.step = step
        # 隐藏层的维数
        self.hide_size = hide_size
        # 批量
        self.batch_size = batch_size
        # 会话的最后一维。是预测的目标
        self.tar = tf.placeholder(dtype=tf.int32, shape=[batch_size, ], name='target')
        self.W = tf.get_variable(name='W', initializer=tf.random_normal_initializer(0, 0.1),
                                 shape=[self.item_num, self.item_num], dtype=tf.float32)
        self.f_embedding = tf.concat([self.embeddings, self.embedding_feat()], axis=1)
        self.in_embedding = self.embedding_suit()
        self.alpha = alpha
        self.beta = beta

    def forward(self, re_embeddings, train=True):
        """
        todo:GRU把代码库表示之后的操作，包括损失函数等。。。
        得出的向量，与所有代码库的向量进行交叉乘，获取概率，这里还不能降维
        :param re_embeddings:
        :return:
        """
        u_embeddings = tf.stack(
            [tf.nn.embedding_lookup(re_embeddings[i], self.n_session[i] - 1) for i in range(self.batch_size)])
        logits = tf.matmul(tf.reshape(u_embeddings, [-1, self.out_size]), self.in_embedding, transpose_b=True)
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar, logits=logits))
        # 查找所有变量的名称
        self.vars = tf.global_variables()

        if train:
            # L2正则化,不对这几个参数进行正则化
            lossL2 = tf.add_n(
                [tf.nn.l2_loss(re_embeddings), tf.nn.l2_loss(self.f_embedding), tf.nn.l2_loss(self.emb_smaller)])
            #lossL2=sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = self.alpha * loss + self.beta * lossL2
            # 训练的是正则化参数
            tf.summary.histogram('loss', loss)
            return loss, tf.nn.softmax(logits)
        else:
            return loss, tf.nn.softmax(logits)

    def embedding_feat(self):
        """
        给向量进行嵌入。。。
        :return:
        """
        self.f_w = tf.get_variable(name='f_w', shape=[self.f_dim, self.f_out_size],
                                   dtype=tf.float32, initializer=tf.random_normal_initializer(-0.01, 0.01))
        self.f_b = tf.get_variable(name='f_b', shape=[self.f_out_size], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(-0.01, 0.01))
        f_emb = tf.matmul(self.other_feat, self.f_w)
        return tf.nn.sigmoid(f_emb)

    def embedding_suit(self):
        """
        节点向量的维度适应gru的输入
        :return:
        """
        self.emb_smaller = tf.get_variable(name='small', shape=[self.dimension + self.f_out_size, self.hide_size],
                                           dtype=tf.float32, initializer=tf.random_normal_initializer(-0.01, 0.01))
        self.emb_smaler_b = tf.get_variable(name='smaller_b', shape=[self.hide_size], dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(-0.01, 0.01))
        return tf.sigmoid(tf.matmul(self.f_embedding, self.emb_smaller) + self.emb_smaler_b)

    def run(self, fetches, tar, item, embeddings, feat, n_session):
        """
        todo:参数可能不够，还要添加
        :param fetches:
        :param tar:
        :param item:
        :param embeddings:
        :return:
        """
        return self.sess.run(fetches, feed_dict={self.tar: tar,
                                                 self.embeddings: embeddings, self.item: item,
                                                 self.n_session: n_session, self.other_feat: feat})


class GGNN(Model):
    """
    todo:没设置额外的超参数。。。
    """

    def __init__(self, dimension, f_dim, f_out_size, item_n, hidden_size=120, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, alpha=0.1, beta=0.2):
        super(GGNN, self).__init__(dimension, f_dim, f_out_size, item_n, batch_size, hidden_size, out_size,
                                   step=step, alpha=alpha, beta=beta)
        # 初始化用户的表示，这也是待推荐的用户的向量表示。。
        # 图的节点个数
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        with tf.variable_scope('ggnn_model', reuse=tf.AUTO_REUSE):
            # forward这这里调用了
            self.loss_train, self.logits = self.forward(self.gnn())
        with tf.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.test_logits = self.forward(self.gnn(), train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step,
                                                        decay_steps=decay, decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_train,
                                                                                  global_step=self.global_step)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        writer = tf.summary.FileWriter('logs/', self.sess.graph)
        self.sess.run(init)

    def gnn(self):
        fin_state = tf.nn.embedding_lookup(self.in_embedding, self.item)
        fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.hide_size])
        u = tf.zeros(shape=tf.shape(fin_state))
        cell = tf.nn.rnn_cell.GRUCell(self.hide_size, reuse=tf.AUTO_REUSE, name='cell')
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            for i in range(self.step):
                state_out, fin_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(u, [-1, self.out_size]),
                                                                              axis=1),
                                                         initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        self.u = state_out
        return tf.reshape(state_out, [self.batch_size, -1, self.out_size])

    def save(self):
        print("保存模型")
        saver = tf.train.Saver()
        saver.save(self.sess, 'my-model')
