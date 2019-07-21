# -*- coding: utf-8 -*-
"""
 Module Summmary: 用tf实现简单的FFM，基本思路是改写tf-fm，field在计算交叉项时加上
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/07/20
"""

class FFM:
    """
    实现FFM算法，总体思路沿用FM
    """
    def __init__(self, batch_size, learning_rate,
                 data_path, field_num,
                 feature_num, feature2field, data_set):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.data_path = data_path  # 训练文件的路径
        self.field_num = field_num
        self.feature_num = feature_num
        self.feature2field = feature2field # feature: field 对应的字典
        self.data_set = data_set # libfm 训练集
        #### input ####
        with tf.name_scope('input'):
            self.label = tf.placeholder(tf.float32, shape=(self.batch_size))
            self.feature_value = []
            for idx in xrange(0, feature_num):
                self.feature_value.append(
                    tf.placeholder(tf.float32,
                                   shape=(self.batch_size),
                                   name='feature_{}'.format(idx)))
        #### embedding ####
        with tf.name_scope('embedding_matrix'):
            # W 参数
            self.liner_weight = tf.get_variable(name='line_weight',
                                                shape=[feature_num],
                                                dtype=tf.float32,
                                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.field_embedding = [] # 存储按filed划分的embedding向量
            for idx in xrange(0, self.feature_num):
                self.field_embedding.append(tf.get_variable(name='field_embedding{}'.format(idx),
                                                            shape=[field_num],
                                                            dtype=tf.float32,
                                                            initializer=tf.truncated_normal_initializer(stddev=0.01)))
        #### output ####
        with tf.name_scope('output'):
            self.b0 = tf.get_variable(name='bias_0', shape=[1], dtype=tf.float32) # b 参数
            # 线性部分
            self.liner_term = tf.reduce_sum(tf.multiply(tf.transpose(
                tf.convert_to_tensor(self.feature_value),perm=[1, 0])
                , self.liner_weight))
            # 交叉部分
            self.qua_term = tf.get_variable(name='quad_term', shape=[1], dtype=tf.float32)
            for f1 in xrange(0, feature_num - 1):
                for f2 in xrange(f1 + 1, feature_num):
                    # 与FM不同的地方，不同的field对应不用的隐变量
                    W1 = tf.nn.embedding_lookup(self.field_embedding[f1], self.feature2field[f2])
                    W2 = tf.nn.embedding_lookup(self.field_embedding[f2], self.feature2field[f1])
                    self.qua_term += W1 * W2 * self.feature_value[f1] * self.feature_value[f2]
            # 输出 sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j) + b0
            self.predict = self.liner_term + self.qua_term + self.b0
            # 计算损失
            self.losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.predict))
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam')
            self.grad = self.optimizer.compute_gradients(self.losses)
            self.opt = self.optimizer.apply_gradients(self.grad)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.loop_step = 0


    def step(self):
        """
        训练一个epoch
        输出：
            loss
        """
        self.loop_step += 1
        feature, label = self.get_data()
        # feed 特征和label
        feed_dict = {}
        feed_dict[self.label] = label
        arr_feature = np.transpose(np.array(feature))
        for idx in xrange(0, self.feature_num):
            feed_dict[self.feature_value[idx]] = arr_feature[idx]
        _,summary, loss_value = self.sess.run([self.opt,self.merged, self.losses], feed_dict=feed_dict)
        self.writer.add_summary(summary, self.loop_step)
        return loss_value

    def get_data(self):
        """
        获取每一轮需要的数据
        输出：
            feature  list
            label  list
        """
        feature = []
        label = []
        # 获取一个batch_size的数据
        for x in xrange(0, self.batch_size):
            t_feature = [0.0] * feature_num
            sample = self.data_set[random.randint(0, len(self.data_set) - 1)]
            label.append(sample[-1])
            sample = sample[:-1]
            for f in sample:
                t_feature[int(f.split(':')[0])] = float(f.split(':')[1])
            feature.append(t_feature)
        return feature, label


def prepare_data(file_path=data_path):
    """
    预处理数据，将libffm的数据格式，改写转化成libfm格式，同时记录feature对应的field
    输入：
        file_path 文件位置 str
    输出：
        data_set 输入模型的样本([{feature, value},..., lable])  list
    """
    feature2field = {} # 记录feature所处的field
    data_set = [] # 模型需要的样本格式
    global field_num
    global feature_num
    # 按行解析libffm格式的训练文件
    for sample in open(file_path, 'r'):
        sample_data = []
        field_features = sample.split()[1:]
        for field_feature_pair in field_features:
            # 拆分某一行的所有的feature和field
            feature = int(field_feature_pair.split(':')[1])
            field = int(field_feature_pair.split(':')[0])
            value = float(field_feature_pair.split(':')[0])
            # 编号从0开始，计数从1开始， 注意变换
            if (field + 1 > field_num):
                field_num = field + 1
            if (feature + 1 > feature_num):
                feature_num = feature + 1
            feature2field[feature] = field
            sample_data.append('{}:{}'.format(feature, value))
        sample_data.append(int(sample[0]))
        data_set.append(sample_data)
    return data_set, feature2field

if __name__ == "__main__":
    print "Alone - Alan Walker/Noonie Bao"
    batch_size = 128
    learning_rate = 0.001
    data_path = './data/test_data.txt' # libffm格式的文件
    # 预定义特征数和field数，属于全局变量
    field_num = 0
    feature_num = 0
    data_set, feature_map = prepare_data(file_path=data_path) # 获取libfm和feature:field字典
    print "feature num {} field num {}".format(feature_num, field_num)
    ffm = FFM(batch_size, learning_rate, data_path, field_num, feature_num, feature_map, data_set)
    # 训练并计算loss
    for loop in xrange(0, 1000):
        losses = ffm.step()
        if (loop % 50):
            print "loop:{} losses:{}".format(loop, losses)


