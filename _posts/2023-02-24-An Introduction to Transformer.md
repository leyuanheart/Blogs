---
layout:     post
title:      An Introduction to Transformer
subtitle:   
date:       2023-02-24
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Attention
    - Transformer
    - Pytorch
---

开始写这篇博客的时间是2023年2月8日，此时ChatGPT已经爆火了一段时间，并且丝毫没有热度下降的情况，大有一飞冲天之势。所以我感觉有必要去深入学习一下它相关的知识，以后大概率会用得上。

ChatGPT是由InstructGPT发展而来，它们的基础都是GPT，而GPT的核心结构就是Transformer。因此，先学Transformer！

说到Transformer，就不得不提一篇大名鼎鼎的论文 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)。论文为了解决sequence transduction（通俗理解就是语言翻译）问题，首次提出了只由Attention结构组成，完全不使用RNN或者CNN结构的Transformer。可以说Transformer的核心就是Attention，而且Attention并不是仅能用于语言翻译，还可以用于其他的自然语言处理（NLP）问题，比如文本生成（GPT）。

如果你之前完全没有接触过NLP这一方面的内容，可能你会觉得上面说的很绕。我希望通过这篇文章能让大家对Attention、Transformer以及GPT的关系有一个更深入的理解。综上，我们首先要学习的就是Attention。

## Attention

Attention也称为注意力机制，我们可以从名称上来对其有一个直观的认识。我们就以人的注意力来说明，比如说我们看这张最近很火的电视剧《狂飙》的海报。我们首先可能看个整体，知道这上面有6个人物，然后我们的注意力会聚焦到自己最感兴趣的1个或者2个人物上，仔细的观看他们身上的一些细节，对其他人物的注意力就没有那么多了，在细品完感兴趣的人物之后，可能再会把注意力转移到其他人物上。这里想表达的内容有两点：

1. 我们对不同部分的关注度（权重）可能是不同的。
2. 当我们将注意力聚焦到不同的部分，可以得到不同的信息。

<img title="" src="https://pic.imgdb.cn/item/63f83f0df144a01007e281a4.jpg" alt="" width="289" data-align="center">

接下来我们再看看在NLP中，Attention机制又表示什么？
借用一张知乎上的图：

<img title="" src="https://pic.imgdb.cn/item/63f83f0cf144a01007e280d3.jpg" alt="" data-align="center">

从上图中的这句话，我们可以发现，”吃“和”苹果“之间有着很强的关联性，因此，我们会希望在处理”吃“这个单词时，能够加入”苹果“这个词的信息，从而能够帮助模型更好地理解”吃“这个概念。同理，”绿色的“和”苹果“之间也是有着较大的联系，在处理”绿色的“时候也应该包含一些”苹果“的信息，这其实就是我们通常说的”上下文“信息。Attention机制可以帮助我们在处理单个token的时候，引入上下文信息。那么到底是怎么做的呢？

### Self-Attention

接下来我们先介绍Self-Attention，它可以看成是Attention的简化版，之所以带个self是因为它的计算完全只使用了输入本身的信息。

Self-Attention是一个序列到序列的操作，输入一个向量序列$(x_1,\dots,x_n)$，输出一个向量序列$(z_1,\ldots,z_n)$。我们以上面的那句话为例来说明如何计算。假设我们把每个单词都用一个$d_e$维向量表示（这个是所谓的embedding），那么我们的输入就是$(x_1,\dots,x_6)$。现在要考虑上下文信息，那么对于每一个$z_i$，可以考虑一个加权平均，即$z_i=\sum_j w_{ij}x_j$，其中权重$w_{ij}=f(x_i,x_j)$且$\sum_j w_{ij}=1$。权重的大小取决于两个词之间的联系的强弱，最常见的做法就是计算內积再归一化，即

$$
\begin{align*}
w'_{ij} & = x_i^Tx_j, \\
w_{ij} & = \frac{\exp w'_{ij}}{\sum_j \exp w'_{ij}},~ \text{ for } i =1,\ldots,6.
\end{align*}
$$

以上就是Self-Attention最基本的操作了。可以看到，这里面没有任何需要学习的参数，对于一个给定的输入，它的输出就是确定的了。为了使整个过程更灵活，对于一个输入$x$，我们分别为其生成3个向量：Query (q)、Key (k)和Value (v)。这3个向量是抽象的概念，<mark>之后会通过训练得到</mark>，但是可以从它们的命名上大致理解一下各自的作用。这种命名方式来源于搜索领域。假设我们有一个Key-Value形式的数据集，就比如说一个论文库，它的关键词就可以作为key，论文内容就是value。那现在用户有一个搜索请求（query），搜索引擎希望能够返回和query最相关的key对应的value。

对应到我们这里，假设我们通过$(x_1,\dots,x_6)$分别得到$(q_1,\dots,q_6)$，$(k_1,\dots,k_6)$和$(v_1,\dots,v_6)$，其中$q,k,v$的维数可以自定，但必须满足$q$和$k$的维数相同，这里为了叙述方便，先统一都设为$d_e$。那么

$$
\begin{align*}
z_i & = \sum_j w_{ij} v_j, \\
w'_{ij} & = \frac{q_i^Tk_j}{\sqrt{d_e}},  ~~~~~~~~~\text{  for } i =1,\ldots,6 \\
w_{ij} & = \frac{\exp w'_{ij}}{\sum_j \exp w'_{ij}}.
\end{align*}
$$

以$z_1$的计算为例，大致的流程如下图所示。

![](https://pic.imgdb.cn/item/63f83f0df144a01007e280f6.jpg)

### Scaled Dot-Product Attention

这里除以$\sqrt{d_k}$的原因是为了控制计算出来的权重的方差。假设$q$和$k$中的每个元素都是从标准正态分布中随机产生，那么$w'=q^Tv$的方差就是$d_k$。如果方差很大，那么$w$中可能出现一些极大或者极小的数，而softmax函数在这些区域的梯度都很小，不利于训练；另外如果出现非常大的数，也会使得计算出来的w变成一个Dirac分布（$\lim_{x \to \infty} \text{softmax}\{0, x\} = \max \{0, x\}$）。

在实际应用中，我们不会一个一个计算$z$，而是使用矩阵计算，公式如下

<img title="" src="https://pic.imgdb.cn/item/63f83f0df144a01007e28154.png" alt="" data-align="center" width="349">

其中$Q=[q_1 \ldots q_6]^T$，$K=[k_1 \ldots,k_6]$，$V=[v_1 \ldots v_6]^T$。图解如下：

<img title="" src="https://pic.imgdb.cn/item/63f83f0df144a01007e28184.jpg" alt="" data-align="inline" width="717">

### Multi-Head Attention

我们在之前提到过，聚焦不同的部分所得到的的信息是不同的。这在NLP问题中也适用，一个词和不同词之间的关系是不一样的，还是拿上面的那句话为例，“苹果”和“吃”可以认为是谓语动词+宾语的关系，而“苹果”和“绿色的”可以是形容词+名词的关系。因此，我们往往是利用多个Attention结构去分别学习不同的关系。

之前是一个维度为$d_e$的Attention，现在考虑$h$个Attention，同时为了保持计算量差不多，这里还要做一些修改。我们令$d_k$表示<mark>单个Attention中</mark>query和key的维数（因为要做內积，两者的维数必须相同），$d_v$表示<mark>单个Attention中</mark>value的维数。之前只有一个Attention的时候，我们是令$d_k=d_v=d_e$，现在有$h$个attention，我们就令$d_k=d_v=d_e/h$，同时还要构造投影矩阵$W_i^Q\in \mathbb{R}^{d_e\times d_k}$，$W_i^K\in \mathbb{R}^{d_e\times d_k}$，$W_i^V\in \mathbb{R}^{d_e\times d_v}$和$W_i^O \in \mathbb{R}^{hd_v\times d_e}$计算公式如下图所示：

<img title="" src="https://pic.imgdb.cn/item/63f840b3f144a01007e4fcb7.png" alt="" data-align="inline" width="694">

相应的流程图如下：

<img title="" src="https://pic.imgdb.cn/item/63f840b3f144a01007e4fcbc.jpg" alt="" data-align="inline" width="718">

以上基本上就是关于Attention的比较重要的内容了。注意到上面的queries、keys和values全部都是来$x$，所以严格意义上来讲应该叫Self-Attention。而Attention就可以更灵活一些，queries、keys和values可以来自不同的数据，就比如Transformer中除了在encoder和decoder中各自使用了Self-Attention之外，为了在两个部分之间进行信息传递，使用了Attention：keys和values由encoder产生，而queries来自decoder。encoder和decoder在翻译问题中分别接收两种不同的语言，而在文本-图片转化问题中，可以分别接收图片和文字。因此，不同的模态都可以通过Attention连接起来，使得Attention的用途十分广泛。

### 总结

3句话：

- Attention的核心是将Query-Key-Value结构用神经网络表示出来（涉及到如何构造这3个部分，以及如何计算权重等，个人理解）。

- Self-Attention更多的是一个特征提取器，通过考虑原始特征之前的关系来构造出更好的特征表示。

- Attention可以连接两种不同的模态，从而可以处理很多问题。

## Transformer

 下面我们就开始介绍Transformer了。下图是论文中Transformer的框架，里面的部分我们一个一个来讲。

<img title="" src="https://pic.imgdb.cn/item/63f840b4f144a01007e4fcd7.png" alt="" data-align="inline">

首先这里面的Scaled Dot-Product Attention和Multi-Head Attention之前已经介绍了，只不过前者多出了一个“Mask（可选项）”，这个我们会在后面介绍。然后，我们回顾Transformer的老本行：翻译任务。它的大框架分成两部分：encoder和Decoder。Encoder的目的是生成源语言的一种中间表示，Decoder的目标是对这种中间表示进行解码，生成目标语言。大致可以看到，Encoder和Decoder的结构比较相似，所以我们先重点分析一下Encoder，之后对于Decoder就只介绍与Encoder不一样的地方。

### Encoder

我们可以看到图中有一个$N\times$的符号，这个表示的是方框里的结构被重复使用了$N$次。我们通常把这种结构成为block，中文翻译成模块。模块给人的感觉就是可拆卸、可组装、可重复利用。我们先来介绍这个模块，里面包括几个重要的部分：

- Multi-head Self-Attention

- Residual Connection

- Layer Normalization

- Feed-Forward Network

接下来分别介绍一下这几个部分。

#### Residual Connection

残差神经网络最早见于 [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)。它的想法非常直观，就是在原本的网络结构中再加上一条直接的从输入到输出的连接，见下图：

<img title="" src="https://pic.imgdb.cn/item/63f840b4f144a01007e4fcfd.jpg" alt="" width="255" data-align="center">

这样做可以避免神经网络在训练时出现梯度消失的问题，从而可以训练非常深的神经网络模型。

#### Layer Normalization

Layer Normalization来自于这篇文章 [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)。它和Batch Normalization功能相似（稳定权重的分布，起到正则化的作用，加速训练）但做法略微不同。最直观的理解是Batch Normalization是列标准化，Layer Normalization是行标准化。

<img title="" src="https://pic.imgdb.cn/item/63f840b4f144a01007e4fd24.jpg" alt="" width="448" data-align="center">

可能有人会说，Batch Norm好理解，就是把一批数据的每一个维度的样本给标准化一下，可是Layer Norm把一个样本的所有维度给标准化，这种做法似乎没什么道理，比如一个人的身高、体重、年龄，这三个特征的量纲都不同，有啥好标准化的。但其实这个问题在图像和文本应用中就不是一个问题了，因为它们的样本要么就都是像素点，要么就都是字符或者词语，可以认为量纲都是一样的，也就可以做标准化。因此，大部分网上介绍这些正则化都是用3维图形来示例的，比如这样

<img title="" src="https://pic.imgdb.cn/item/63f84168f144a01007e5ded5.jpg" alt="" data-align="center" width="420">

而且在NLP问题中，Layer Norm反而比Batch Norm应用更广泛。因为从上图上来看，Batch Norm是按照每句话的位置将数据进行标准化，但是语言是很复杂的，不同位置的词的差异性是很大的，标准化反而可能会损害模型的表现；而Layer Norm是将每一句话进行标准化，相对来说更合理。

现在已经有许多各种各样的Normalization，感兴趣的可以看看这个博客 [深度学习中的Normalization方法](https://www.cnblogs.com/LXP-Never/p/11566064.html)。

#### Position-wise Feed-Forward Network

在Encoder中，经过了Self-Attention之后，在输出之前还要在接一个全连接层，使用Relu激活函数，直接上图

<img title="" src="https://pic.imgdb.cn/item/63f84168f144a01007e5dee0.png" alt="" data-align="center">

值得注意的是，对于一句话中的每一个位置是单独使用了一个FFN的，所以称为Positional-wise。借用知乎上的一张图。但其实每个MLP的权重都是一样的，所以本质上还是只使用了一个FFN。

![](https://pic.imgdb.cn/item/63f84168f144a01007e5def2.jpg)

至此，$N \times$框中的内容就介绍完了，接着我们往下看，会发现一个叫Positional Encoding的东西，为啥会有这个呢？这其实是由于Attention机制本身并没有记录与位置相关的信息导致的。

在RNN里，$t$时刻的输入是带有一部分前$t-1$时刻的信息的，因此它天然就是有顺序的。但是Attention不是这样，回顾一下之前的计算，我们发现每一个位置都是与其他所有位置计算权重再加权平均，它们是可以并行计算的，而且打乱词的顺序，得到对应的输出也是一样（这里说的更加严谨一点应该是，打乱key-value的顺序，用同样的$q$得到的$z$是一样的）。因此，Attention不具有位置概念，任意两个词之间的距离都是相等的。那在NLP中，语序毫无疑问是一个重要的特征，因此得想办法把这部分信息引入。

如何引入位置信息呢，做法很直接，我们可以创建一个和输入等长的向量序列，每个向量的维数也和输入的维数相同，这些向量包含位置信息，然后和输入序列相加，从而得到包含位置信息的输入。方法有两种，第一种是**Positional Embedding**。比如，还是以“She is eating a green apple.”为例，假设每个词对应的embedding（在之后会介绍）是$d_e$维，即我们有$(x_1,\ldots,x_6)$，其中$x_i \in \mathbb{R}^{d_e}$。那么我们创建位置向量序列$(v_1,\ldots,v_6)$且$v_i \in \mathbb{R}^{d_e}$。然后通过模型来学习出这些向量到底是什么。但这种方法有一个问题，就是模型在预测阶段只能预测在训练是见到过的位置向量，针对在这个例子，通俗点讲就是一旦碰到句子长度超过6的，模型就不work了。

另一种就是接下来要介绍的Positional Encoding。

#### Positional Encoding

Encoding不进行训练，而是用一个给定的函数来编码位置信息，然后希望模型能通过学习掌握到使用这些信息的方法。这种方法就不会出现上面提到的问题，因为对于任意位置，encoding都能给出一个向量，这样模型至少是能正常进行预测的。Transformer使用的是这种方法，论文中使用的函数如下图所示：

<img title="" src="https://pic.imgdb.cn/item/63f84168f144a01007e5df0f.png" alt="" data-align="center">

其中*pos*表示词所处的位置，$i$表示词向量的第$i$维，$d_{model}$就是我们这里的$d_e$，表示词向量的维数。下面是一个小例子，假设词向量是4维的，那么

![](https://pic.imgdb.cn/item/63f84169f144a01007e5df28.png)

我们也可以看一下不同的维度$i$上，不同位置的编码是一个怎样的结果，下图是一个示例![](https://pic.imgdb.cn/item/63f841cbf144a01007e65c8d.png)

其实我是觉得这个编码函数还是挺妙的。因为首先它是满足两个位置的向量做內积，其大小和位置的距离是成反比的，下图是一个例子：

![](https://pic.imgdb.cn/item/63f841cbf144a01007e65cf5.jpg)

另外，它不仅能表示绝对位置，还可以表示相对位置。三角函数为什么能捕捉到相对位置信息依赖于以下两个公式：

$$
\begin{gather*}
\sin(\alpha+\beta)=\sin(\alpha)\cos(\beta)+\cos(\alpha)\sin(\beta) \\
\cos(\alpha+\beta)=\cos(\alpha)\cos(\beta)-\sin(\alpha)\sin(\beta)
\end{gather*}
$$

对于单词序列的位置偏移$k$，$PE_{(pos+k)}$可以表示成$PE_{(pos)}$的一个线性函数：

$$
\begin{gather*}
PE_{(pos+k,2i)}=PE_{(pos,2i)} \times PE_{(k,2i+1)}+PE_{(pos,2i+i)}\times PE_{(k,2i)} \\
PE_{(pos+k,2i+1)}=PE_{(pos,2i+i)}\times PE_{(k,2i+1)} - PE_{(pos,2i)}\times PE_{(k,2i)}
\end{gather*}
$$

#### Input Embedding

最后我们来介绍一下Input Embedding。对于NLP问题，我们最原始的数据是很多句话。我们首先要做的是Tokenization，中文翻译叫做分词，不过对于英文来说，有时候也不一定是根据单词来进行划分，也有可能是根据字符来划分，还有可能是通过sub-word来划分，分出来的一个个部分我们称其为token。这一块具体怎么用我也不太清楚，感兴趣的可以看看这篇博客 [NLP中的Tokenization](https://zhuanlan.zhihu.com/p/444774532)。我们这里就按照分词的概念来做，还是拿“She is eating a green apple.”为例，按词来Tokenize得到的是【She, is, eating, a, green, apple, `<eos>`】。`<eos>`是指end of sequence，也算一个token。计算机是无法直接读取这些token的，所以我们需要对每个token进行编码，这个过程就称为Embedding。再进一步解释，就是我们要为每一个token找到一个embedding向量，使其能够表示这个token。我们拿电影来类别，比如我们要描述一部电影，我们可以从几个维度去描述，比如它属于动作片的部分占多少，属于爱情片的部分占多少，包含喜剧的成分有多少。这些维度是我们自己拍脑袋想出来，对于NLP中的embedding，我们先确定向量的维数，然后通过模型学习出这些embedding vector是什么。当然也可以考虑用其他人训练好的embeddings，Transformer论文中使用的前者。

好了，Encoder部分就介绍的差不多了，下面我们来看Decoder。

### Decoder

Decoder和Encoder的结构大致相同，但是也有一些差异，主要体现在一下3个方面：

- 在Decoder内的Self-Attention使用了Sequence Mask机制。

- 多出了一个和Encoder连接的Attention。

- 在最后输出时增加了一个Linear和Softmax层来预测每个token对应的概率。

在介绍以上几个部分之前，我们先来看看Decoder的工作流程。我们用一个例子来说明。比如有一个翻译任务：`“我喜欢蛋糕”`$\to$`I like cakes`。

假设Encoder已经输入了`“我喜欢蛋糕”`，然后输出了信息，我们称为Encoder Embedding。对于Decoder来说，我们首先要给一个起始符`<s>`作为Decoder的初始输入。

```
Time Step 1:
    原始输入: 输入 <s>
    中间输入: Encoder Embedding
    产生预测: I
Time Step 2:
    原始输入: <s> + I
    中间输入: Encoder Embedding
    产生预测: like
Time Step 3:
    原始输入: <s> + I + like
    中间输入: Encoder Embedding
    产生预测: cakes
```

通俗来说，Decoder是一个单词一个单词来解码，最后输出一整个句子的。

#### Masked Attention

其实在Transformer中有两种Mask机制，一种是Decoder中独有的Sequence Mask，另一种是所有Attention都会用到的Padding Mask。我们先介绍第一种。

##### Sequence Mask

Sequence Mask的目的是让Decoder不能看见未来的信息。这直观上很好理解，我们在$t$时刻进行解码时，只能依赖于$t$之前的信息。做法是这样的：假设句子的长度为6。回顾一下之前Attention中计算权重的公式$Softmax(\frac{QK^T}{\sqrt{d_e}})$，其中$QK^T \in \mathbb{R}^{6 \times 6}$。我们生成一个$6 \times 6$维的下三角矩阵$M$：

```
[[1, 0, 0, 0, 0, 0]
 [1, 1, 0, 0, 0, 0]
 [1, 1, 1, 0, 0, 0]
 [1, 1, 1, 1, 0, 0]
 [1, 1, 1, 1, 1, 1]]
```

然后将M矩阵为0的部分对应的$QK^T$中的元素设置成`-inf`，这样再经过Softmax之后结果就是0。（具体实现过程在之后的代码部分会介绍）

看到这里不知道大家是不是有一个疑问，就是按照之前介绍的Decoder的流程，它就是预测出一个词，把其加入新的输入中，生成下一个词。这本身就不会用到未来的信息，那为什么还要有Mask的操作呢？这其实也是我一开始有的疑问，后来再研究了一下，发现了原因。我们上面描述的Decoder的流程是指它在Inference阶段的行为，而在Training阶段，它的输入是已经翻译好的一整个句子，在第$t$步只能用到$t$之前的词，Decoder上一步输出的内容并不会作为新的输入，举个例子，比如第一步Decoder输入`<s>`，得到预测`he`，然后在下一步，并不是用`he`而是用真实的结果`I`作为输入。<mark>说实话，这个是我没想到的训练方式，要是我自己训练，应该就会使用上一步的输出了</mark>。

##### Padding Mask

Padding Mask的作用是使得每一批次序列长度保持一致。因为句子有长有短，所以我们需要先设置一个固定的序列长度$d_l$，如果句子长度小于$d_l$，就在后面补0（padding），如果句子过长，就只截取前$d_l$个token。Padding Mask在所有的Scaled Dot-Product Attention中都要用到，处理方法也和Sequence Mask一样，在需要padding的地方赋值一个`-inf`，这样经过Softmax之后权重就为0了，也就是让Attention不需要去考虑这些位置。

#### Connection with Encoder

Decoder和Encoder的第二处不同就是它比Encoder多了一个Attention部分，这个部分是用来将Encoder的信息传输给Decoder的（不然Encoder搞了半天也没啥用）。具体的做法就是将Encoder的输出（z）做一些变换后得到Key-Value，然后Decoder的输入（x）做一些变换作为Query，通过Attention将它们结合起来，即：

$$
q = x^T W^Q, ~~~ k = z^T W^K, ~~~ v = z^T W^V.
$$

![](https://pic.imgdb.cn/item/63f841cbf144a01007e65d1a.png)

<mark>注意：Encoder输出的信息会输入到所有$N$个Decoder Block中。</mark>

#### The Final Linear and Softmax Layer

要想得到单词的预测，Decoder最后是通过一个Linear Layer将输出扩展到与vocabulary size一样的维度，然后作用一个softmax得到每个词的概率，选择概率最大的那一个。

![](https://pic.imgdb.cn/item/63f841ccf144a01007e65d23.png)

我觉得Transformer比较重要的知识点就介绍完了，接下来论文里还介绍了一些在训练过程中用到的小技巧。

### Training

训练的损失函数是交叉熵损失。举个例子来说明一下。假设vocabulary里只包含6个token：【a, am, i, thanks, student, `<eos>`】，我们使用one-hot编码

<img title="" src="https://pic.imgdb.cn/item/63f841ccf144a01007e65d6a.png" alt="" data-align="center" width="257">

Decoder在每一个位置都会输出一个向量，然后和目标向量计算交叉熵，

<img title="" src="https://pic.imgdb.cn/item/63f842fdf144a01007e7d96e.png" alt="" data-align="center" width="283">

模型的总loss是所谓位置的交叉熵的加和。

#### Optimizer

![](https://pic.imgdb.cn/item/63f842fdf144a01007e7d99d.png)

论文里给出了使用的优化器和学习率，需要注意的是，这里的设置很关键，如果不这么设置很可能训不出来。

#### Regularization

论文中使用了两类正则化方法。一类是Dropout，一类是Label Smoothing。

##### Residual Dropout

![](https://pic.imgdb.cn/item/63f842fdf144a01007e7d9a7.png)

这里的意思是说Dropout用在了两类地方：

1. 在每一层sub-layer输出上，在图中可以理解为在每一个`Add & Norm`的下面。

2. 在input embedding和positional encoding加和之后，在图上可以理解为$\oplus$的上面。

3. 在代码中多出来的第一处，是在Scaled Dot-Product Attention中，用softmax算出权重后，对权重用了一次Dropout。

4. 在代码中多出来的第二处，是在Position-wise Feed-Forward Network中，经过第一个隐层后（线性+激活函数）用了一次Dropout。

##### Label Smoothing

设token所在的真实位置为$y$，一共有$K$个位置。传统的one-hot encoding是这样的：

$$
P_i = \begin{cases}
1, & \text{if } i = y \\
0, & \text{if } i \neq y
\end{cases}
$$

而Label Smoothing（$\epsilon_{ls}=0.1$）是这样的：

$$
P_i = \begin{cases}
1-\epsilon_{ls}, & \text{if } i = y \\
\frac{\epsilon_{ls}}{K-1}, & \text{if } i \neq y
\end{cases}
$$

### 代码

完整的代码我放在这个github仓库里了：[An-Implementation-of-Transformer](https://github.com/leyuanheart/An-Implementation-of-Transformer)

## 参考资料

[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[Self-Attention和Transformer](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#skip-connection-he-layer-normalization)

[Transformer10个重要问题](https://mp.weixin.qq.com/s/OSTJx-JSpoeMVEKZRjVZXA)

[Transformer 知识点理解](https://zhuanlan.zhihu.com/p/58009338)

[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/#attention-visualization)

~~[nanoGPT代码~~](https://github.com/karpathy/nanoGPT)

~~[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) （如果不能看YouTube，B站上也有）~~
