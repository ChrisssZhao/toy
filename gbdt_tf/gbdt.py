# -*- coding: utf-8 -*-
"""
 Module Summmary: 使用tensorflow 实现简单的gbdt 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/04/14
"""
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner

#加载数据，使用minst数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

#定义超参数
batch_size = 1024
num_classes = 10 #10个数字
num_features = 784 #图片大小是 28 * 28 = 784
max_steps = 10000

#提升数的参数
learning_rate = 0.1
l1_regul = 0.01
l2_regul = 0.1
examples_per_layer = 1000 #逐层学习的参数，控制样本数量
num_trees = 10
max_depth = 16

#tf gbdt的配置填充
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
learner_config.growing_mode = growing_mode
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = (gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)

#定义分类器
gbdt_model = GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        n_classes=num_classes,
        examples_per_layer=examples_per_layer,
        num_trees=num_trees,
        center_bias=False,
        config=run_config
        )

#mini-batch输入, 训练模型
input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images},
        y=mnist.train.labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
        )
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

#测试
input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images},
        y=mnist.test.labels,
        batch_size=batch_size,
        shuffle=False
        )
res = gbdt_model.evaluate(input_fn=input_fn)
print "acc: ", res['accuracy']

