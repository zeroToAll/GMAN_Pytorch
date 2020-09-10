# GMAN_Pytorch
这是论文[GMAN:A Graph Multi-Attention Network for Traffic Prediction](https://www.aaai.org/ojs/index.php/AAAI/article/view/5477)的pytorch实现，原文提供了TensorFlow的实现，但是对TensorFlow只停留在能看的阶段。用pytorch实现后希望后续能对其有所改进。

## 论文的主要内容

文章以交通数据为研究对象。考虑了各个监测点之间的关系。实现了监测点的时空特征和监测的序列特征的融合。提供了一种全新的看待数据的视角。通常对序列数据建模，例如NLP。看待数据的方式是![](http://latex.codecogs.com/svg.latex?X_i \in R^{ time\_step \times e\_dim})。文章把监测点都放在同一个样本上，所以单个样本的维度变成了$X_i \in R^{N\times time\_step \times e\_dim}$。然后用Attention分别提取时间特征和空间特征。时间维度的Attention matrix大小为$R^{N\times time\_step \times time\_step}$。空间维度的Attention matrix大小为$R^{time\_step\times N \times N}$。

## 优点和不足

优点很明显，可以得到更加丰富的特征。缺点就是慢，两个Attention matrix太大。文中的序列长度(time_step)是12，监测节点数(N)是345。训练一轮要4个多小时。如果序列长度过长，可以考虑用膨胀卷积来提取序列特征。空间特征可以用[Synthesizer: Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743)这个文章的思想来改进。直接学习一个固定的Attention matrix，也可以直接用邻接矩阵代替Attention matrix。

