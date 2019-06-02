# 因子分解机在CTR中的运用

## 一、简介

​		点击率预估算法被誉为镶嵌在互联网技术上的明珠，一直以来对这类问题有着很浓厚的兴趣。在学生时代就接触 CTR 和 CVR 问题机会主要来着 IJCAI2018 阿里妈妈搜索广告大赛和第二届腾讯社交广告大赛，前者预测广告的转化为交易的概率，而后者侧重于相似人群拓展。从任务的角度来考虑，这类问题属于标签极度不平衡的二分类问题，使用的算法也大同小异。从调包开始，我学习这类问题的主要算法路线是 LR -> XGB/LGBM->FM/FFM ->各种 DNN 模型。

​		机器学习都是从简单的贝叶斯和逻辑回归开始，教材翻到最后也找不到 XGB 和 FFM 算法（无论多新的书），确实是件遗憾的事情。LR 之经典和伟大毋庸置疑，到现在仍有很实用的场景，不过似乎大家潜意识都倾向于用听起来比较 fancy 的算法（包括我自己）。在疯狂的调参之后，感觉到 CTR 类算法有两条明显的支线：1.以树模型进化出各种 Boosting 算法；2.以因子分解机为主不断衍生。因子分解机（Factorization Machine）对我来说远没有 Xgboost 和 LightGBM 熟悉，抱着学习的态度，本文主要梳理下在工作和学习中遇到 FM-based 的一些感悟和分享。如有纰漏，敬请指出。

# 二、FM/FFM

## 2.1 因子分解机（FM）

​		可能很多人和我一样最先接触到的并不是 FM，当我还是个孩子的时候，我就问同学或者同学问我，XGB 都这么牛X为啥要用 FM。抱住这种疑问结束了学生生涯，参加工作之后才发现FM的强大和与众不同。在点击率预告的问题中通常包含着大量 ID 类特征，送入树模型中通常需要进行 onehot 编码，当数据量到达千万条以上这将是一个十分耗时和消耗资源的过程。更为重要的是在特征十分稀疏的情况下，LR 和 XGB 等模型很难学习特征之间的交叉信息，这也是因子分解机最主要的目的。

​		简单介绍下因子分解机的原理。我们首先考虑一个多项式模型，并对模型进行二元交叉可以得到下面的式子
$$
f(x)=w_{0}+\sum_{i=1}^{d} w_{i} \cdot x_{i}+\sum_{i=1}^{d} \sum_{j=i+1}^{d} w_{i j} \cdot x_{i} x_{j}
$$
其中 $x$ 表示特征，$d$ 代表特征维度，$w$ 表示系数。很容易得出参数$$w_{i j}$$ 的个数为$\frac{d(d-1)}{2}$ ，当特征维度 $d$ 很大时参数矩阵 $$\left\{w_{i, j}\right\}$$ 几乎不可计算。思考下原因，在多项式模型中 $$w_{i, j}$$ 代表的是两个特征之间的系数，在特征十分稀疏（大部分 $x$ 的值为0）的情况下直接学习参数效率低下。FM提出用 $k$ 维隐向量作为来表示特征，这 $k$ 个值都是表示特征的因子，因此被称为因子分解机，其公式入下所示
$$
f(x)=w_{0}+\sum_{i=1}^{d} w_{i} \cdot x_{i}+\sum_{i=1}^{d} \sum_{j=i+1}^{d}\left(v_{i} \cdot v_{j}\right) \cdot x_{i} x_{j}
$$
其中 $v$ 就代表 $k$ 维的因子。  这样就将 $W=\left\{w_{i, j}\right\}$ 分解为 $W=V^{T} V$ 的形式，这里 $$
V=\left(v_{1}, v_{2}, \cdots, v_{d}\right)
$$ 就是一个 $$
k \times d
$$ 的矩阵。然后我们惊奇的发现，需要训练的参数个数从 $\frac{d(d-1)}{2}$ 降到 $kd$ 。

## 2.2 FFM

​		首先强推美团技术团队的[《深入FFM原理与实践》][1]和作者的 [Slides][2] ，看完这些就可以了解FFM的基本原理。FFM的全称是Field-aware Factorization Machine，是在因子分解机的基础之上引入了域（filed）的概念。FM 有一个很明显的缺点，它不加区分的对待每一个特征，忽略了某一类特征之间的共性。FFM 认为，由同一个 ID 类特征通过onehot编码产生的特征，或者其他特征变换获得的特征，应该同属于一个特殊的集合域，不同的特征和同一个域关联时需要使用不同的隐向量。假如我们一个有 $d$ 个特征和 $f$ 个域，那么每个特征需要用 $f$ 个隐变量表示，也就是一共有 $ d\times f$ 个隐向量。从 FFM 的公式也可以看出，
$$
f(x)=w_{0}+\sum_{i=1}^{d} w_{i} \cdot x_{i}+\sum_{i=1}^{d} \sum_{j=i+1}^{d}\left(v_{i, f_{j}} \cdot v_{j, f_{i}}\right) \cdot x_{i} x_{j}
$$
相对于 FM 而言，FFM 的设计更为复杂与合理。简单的来说，FM 是两个特征之间的直接交叉，FFM更近一步特征是和域进行交叉。当只有一个域的时候，FFM 等价于 FM，也就是说其实 FM 是 FFM 把特征都归为一个域的特例。

​		FFM 的基本原理就不在赘述，有兴趣的可以好好看看上面推荐的两个链接。下面分享下使用FFM的一些注意事项：1. 归一化、归一化、归一化（重要的事情说三遍！！！），包括样本归一化和特征归一化；2. 特征编号，libffm特征的格式为 field:index:value，有些封装好的模型包field编号从0开始，有些从1开始；3. 可以省略value为0的项，零值特征对模型训练没有任何贡献；4. 推荐使用xLearn，速度十分的快。

## 2.3 xLearn

​		[xLearn][3] 是一个十分有用的机器学习工具包，目前 xLearn 已经集成了三种经典的算法，包括LR，FM和FFM，适用于广告点击率预测、推荐系统等多种场景。相比与liblinear、libfm和libffm这三个工具包，它的优势在于性能好和简单易用。经过测试，xLearn 可以比 libfm 快13倍，比 libffm 和 liblinear 快5倍，同时提供 out-of-core 计算，利用外存计算可以在单机处理 1TB 数据，并且支持分布式训练。另外一个令人欣喜的是，xLearn提供了python接口，调用起来十分的方便。

![image-20190526181153244](/Users/qunzhao/Library/Application Support/typora-user-images/image-20190526181153244.png)

​		以我司（[贝壳找房][4]）100W 左右的小批量原始数据集上测试，其中field数量为34，特征维度为1407926，所用时间和表现性能如下所示。

| 算法     | AUC   | 耗时（s） |
| -------- | ----- | --------- |
| LR       | 0.657 | 8.42      |
| FM       | 0.666 | 9.10      |
| FFM      | 0.663 | 34.84     |
| LightGBM | 0.672 | 68.54     |

 从该数据集上来看，xLearn 和 LightGBM 进行对比来看，xLearn的速度优势十分的明显，但是精度稍弱于后者。xLearn速度快。

# 三、deepFM和xDeepFM

## 3.1 Wide & Deep

​		机器学习的领域中有一些令人惊艳的算法，它们的出现给研究者们带来了新的思路，甚至开拓了一个流派。在CTR任务中，Google于2016年发表的 [Wide and Deep][6] 算法将深度学习应用于 Google Play 的推荐系统中，在行业内引起了不小的轰动。严格意义上来说，W&D 算法和这篇文章主要讲的 FM-based 模型并没有关系，但是它提出来的算法框架值得我们好好研究。目前很多算法，包括后面要提到的 deepFM 和 xDeepFM 都是基于它的算法框架进行改进。

​		介绍框架之前，先介绍下论文中提到的两个十分关键的名词 memorization 和 generalization。简单的来说，memorization 和 generalization 是处理特征的两种方式，memorization 考虑的是如何将原始特征包含尽可能的表达出来，generalization 则是如何泛化学习到原始特征中隐藏的信息。在查阅资料时，发现一个[博文][5]很有意思，文章中有句话很好的解释了 memorization 和 generalization，原文是：The human brain is a sophisticated learning machine, forming rules by memorizing everyday events (“sparrows can fly” and “pigeons can fly”) and generalizing those learnings to apply to things we haven’t seen before (“animals with wings can fly”) 。通常来说，memorization 可以通过线性模型和特征交叉实现，generalization 则需要更多人工的特征工程。

![image-20190602105556943](/Users/qunzhao/Library/Application Support/typora-user-images/image-20190602105556943.png)

​		在回到 Wide & Deep 的框架就比较好理解，它可以分为两个部分，左边 Wide 部分为了memorization，右边 Deep 部分为了generalization。算法的思路是 LR + DNN，将 DNN 的输出和左边的 LR 连接，通过 Sigmoid 层得到输出。结合公式更加清楚，
$$
P(Y=1 | \mathbf{x})=\sigma\left(\mathbf{w}_{w i d e}^{T}[\mathbf{x}, \phi(\mathbf{x})]+\mathbf{w}_{d e e p}^{T} a^{\left(l_{f}\right)}+b\right)
$$
式中，$$\sigma(\cdot)$$ 是 Sigmoid 单元，$$\phi(\mathbf{x})$$ 是对原始特征 $$\mathbf{X}$$ 进行的交叉变换，$$a^{\left(l_{f}\right)}$$ 表示 Deep部分 通过激活函数的最终输出，$$\mathbf{W}$$ 和 $$b$$ 就是常见参数矩阵和偏置。如此看来，W&D 算法并没有想象中的"高深莫测" ，也许这就是生活中不缺少美，只是缺少发现美的眼睛（捂脸.jpg）。

## 3.2 DeepFM

​		计算机行业的飞速发展得益于不断创新和快速迭代。窝工（哈工大）的一位学长在华为诺亚实现期间提出了[DeepFM][7]算法，在 Wide and Deep 的框架基础之上，将因子分解机引入到 Wide 部分。有了之前 Wide and Deep的基础，直接来看 DeepFM 的框架图（论文的图不清楚，找了另一张图）。

![image-20190602122336768](/Users/qunzhao/Library/Application Support/typora-user-images/image-20190602122336768.png)

​		以肉眼可见DeepFM 与  Wide and Deep 最大不同在于，DeepFM 使用了 FM 代替 LR，FM 学习交叉特征而Deep 学习高阶特征。同属于广义线性模型，和 LR 相比 FM 的优势在于可以自动学习特征的交叉，同时又可以处理稀疏的特征，减少了使用 LR 在特征工程上的部分工作量。DeepFM 另一个重要的变化是参数共享，Wide 部分和 Deep 部分都连接在同一个 Embedding 层之后，保证了学习的一致性也提高了模型学习的效率。需要注意的是，FM 层同样用到了原始的稀疏作为输入。论文中还提到，由于DeepFM是端到端的训练，不需要在原始稀疏数据上做任何的人工特征工程。针对这一点，我表示怀疑，算法能学习到的始终是数学表达，脱离真实场景去对数据进行建模并不一定可行，因此针对具体业务逻辑进行人工特征工程是有必要的。

## 3.3 xDeepFM

​		在KDD 2018上提出一个新的模型——极深因子分解机（[xDeepFM][8]），主要是针对 DeepFM 和 [DCN][8] 进行改进。首先，xDeepFM的算法框架仍然沿用 Wide and Deep 的框架，在 Wide 部分加入了作者称之为压缩交互网络（Compressed Interaction Network， 简称CIN）的神经模型结构。先上整体的框架图。![image-20190602143233719](/Users/qunzhao/Library/Application Support/typora-user-images/image-20190602143233719.png)

xDeepFM 仍然沿用了 DeepFM 中 Embedding 共享的思路，文中没有列出的 DCN 也是如此。DCN 的改动是将 DeepFM 中的 FM Layer 替换成 Cross Network 为了学习二阶以上的交叉特征，有兴趣的可以看看原论文。受到 DCN 的启发，总结DCN存在的缺点，xDeepFM 的作者提出了更有效的解决办法。先介绍一下，CIN 和 其他模型不同的是特征交叉为显式的向量级（Vector-wise），而不是隐式的元素级（bit-wise）。举个例子，两个特征向量分别为 $$
(a 1, b 1, c 1)
$$ 和 $$(a 2, b 2, c 2)$$ ，$f$ 是交叉函数，如果交叉的形式如 $$f(w 1 * a 1 * a 2, w 2 * b 1 * b 2, w 3 * c 1 * c 2)$$ 为元素级的，若为 $$f(w *(a 1 * a 2, b 1 * b 2, c 1 * c 2))$$ 则称之为向量级。该作者认为，向量级的交叉特征更符合因子分解机的初衷，特征交互发生在向量级，更兼具记忆与泛化的学习能力。

![image-20190602152242560](/Users/qunzhao/Library/Application Support/typora-user-images/image-20190602152242560.png)

​		下面简单介绍下 CIN，在 CIN 中每一层的神经元都是根据前一层的隐层以及原特征向量推算而来，其计算公式如下：
$$
\mathbf{X}_{h, *}^{k}=\sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} \mathbf{W}_{i j}^{k, h}\left(\mathbf{X}_{i, *}^{k-1} \circ \mathbf{X}_{j, *}^{0}\right)
$$
CIN 的计算主要有两个步骤：(1) 根据前一层隐层状态 $$
\mathbf{X}_{i, *}^{k-1}
$$ 和原始输入数据$$
\mathbf{X}_{j, *}^{0}
$$，计算中间结果$$
Z
$$； (2) 根据中间结果，计算下一层隐层的状态。从上面的图 $c$ 可以看出，其实步骤(1)操作类似于RNN网络，而步骤(2)相当于 CNN 中池化的操作，这样看来 CIN 其实是结合了 RNN 和 CNN 的一种网络结构。最后奉上论文中的实验的效果对比

![image-20190602154225758](/Users/qunzhao/Library/Application Support/typora-user-images/image-20190602154225758.png)



# Reference

[1]: https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html
[2]:https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf
[3]:https://xlearn-doc-cn.readthedocs.io/en/latest/
[4]:https://ke.com

[5]:http://sungsoo.github.io/2017/03/27/wide-and-deep-learning-memorization-generalization-with-tensorflow.htm
[6]: https://arxiv.org/pdf/1606.07792.pdf
[7]:https://arxiv.org/pdf/1703.04247.pdf
[8]: https://arxiv.org/pdf/1803.05170.pdf





