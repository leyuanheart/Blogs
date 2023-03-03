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

ChatGPT是由InstructGPT发展而来，它们的基础都是GPT (Generative Pretrained Transformer)，而GPT的核心结构就是Transformer。因此，先学Transformer！

说到Transformer，就不得不提一篇大名鼎鼎的论文 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)。论文为了解决sequence transduction（通俗理解就是语言翻译）问题，首次提出了只由Attention结构组成、完全不使用RNN或者CNN结构的Transformer。可以说Transformer的核心就是Attention，而且Attention并不是仅能用于语言翻译，还可以用于其他的自然语言处理（NLP）问题，比如文本生成（GPT）。

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

<font color="red">这里我一直有一个疑问，就是所谓的加入上下文信息，就是将其他位置的单词的向量表示乘以一个权重再加到该位置的单词的向量表示上就可以了吗？</font>

我们可以把Self-Attention看成是一种交流机制 (communication mechanism)。如下图所示，把一句话中的每一个词看成一个node，箭头表示信息传递的方向。对于每一个节点，它要做的是汇总所有指向它的节点的信息，做法就是加权平均，权重是data-dependent，通过学习得到。下图是一个有向图的例子，当然对于无向图也是可以的，只要是有边连接都是可以进行communication。

![](https://pic.imgdb.cn/item/6401fc18f144a01007fbaaa3.jpg)

### Scaled Dot-Product Attention

这里除以$\sqrt{d_e}$的原因是为了控制计算出来的权重的方差。假设$q$和$k$中的每个元素都是从标准正态分布中随机产生，那么$w'=q^Tv$的方差就是$d_e$。如果方差很大，那么$w$中可能出现一些极大或者极小的数，而softmax函数在这些区域的梯度都很小，不利于训练；另外如果出现非常大的数，也会使得计算出来的w变成一个Dirac分布（$\lim_{x \to \infty} \text{softmax}\{0, x\} = \max \{0, x\}$）。

在实际应用中，我们不会一个一个计算$z$，而是使用矩阵计算，公式如下

<img title="" src="https://pic.imgdb.cn/item/63f83f0df144a01007e28154.png" alt="" data-align="center" width="349">

其中$Q=[q_1 \ldots q_6]^T$，$K=[k_1 \ldots,k_6]$，$V=[v_1 \ldots v_6]^T$。图解如下：

<img title="" src="https://pic.imgdb.cn/item/63f83f0df144a01007e28184.jpg" alt="" data-align="inline" width="717">

```python
import torch 
from torch import nn

batch_size = 32
seq_len = 6     # squence length
n_embd = 10     # embedding dim (d_e metioned above)
x = torch.randn(batch_size, seq_len, n_embd)

class SelfAttention(nn.Module):
    '''Implement a single head self-attention.'''

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)


    def forward(self, x):
        B, T, d = x.shape   # x shape: [batch_size, max_len, n_embd]

        q = self.query(x)                       # [B, T, head_size]
        k = self.query(x)                       # [B, T, head_size]
        v = self.value(x)                       # [B, T, head_size]

        wei = q @ k.transpose(-2, -1) * d**0.5  # [B, T, T]
        wei = F.softmax(wei, dim=-1) 

        out = wei @ v                           # [B, T, head_size]

        return out


head_size = 8   # dimension of query, key and value (set to be the same by default)        
slf_attn = SelfAttention(head_size)        
print(slf_attn(x).shape)       # [batch_size, max_len, head_size]
```

### Multi-Head Attention

我们在之前提到过，聚焦不同的部分所得到的的信息是不同的。这在NLP问题中也适用，一个词和不同词之间的关系是不一样的，还是拿上面的那句话为例，“苹果”和“吃”可以认为是谓语动词+宾语的关系，而“苹果”和“绿色的”可以是形容词+名词的关系。因此，我们往往是利用多个Attention结构去分别学习不同的关系。

之前是一个维度为$d_e$的Attention，现在考虑$h$个Attention，同时为了保持计算量差不多，这里还要做一些修改。我们令$d_k$表示<mark>单个Attention中</mark>query和key的维数（因为要做內积，两者的维数必须相同），$d_v$表示<mark>单个Attention中</mark>value的维数。之前只有一个Attention的时候，我们是令$d_k=d_v=d_e$，现在有$h$个attention，我们就令$d_k=d_v=d_e/h$，同时还要构造投影矩阵$W_i^Q\in \mathbb{R}^{d_e\times d_k}$，$W_i^K\in \mathbb{R}^{d_e\times d_k}$，$W_i^V\in \mathbb{R}^{d_e\times d_v}$和$W_i^O \in \mathbb{R}^{hd_v\times d_e}$计算公式如下图所示：

<img title="" src="https://pic.imgdb.cn/item/63f840b3f144a01007e4fcb7.png" alt="" data-align="inline" width="694">

相应的流程图如下：

<img title="" src="https://pic.imgdb.cn/item/63f840b3f144a01007e4fcbc.jpg" alt="" data-align="inline" width="718">

以上基本上就是关于Attention的比较重要的内容了。注意到上面的queries、keys和values全部都是来$x$，所以严格意义上来讲应该叫Self-Attention。而Attention就可以更灵活一些，queries、keys和values可以来自不同的数据，就比如Transformer中除了在encoder和decoder中各自使用了Self-Attention之外，为了在两个部分之间进行信息传递，使用了Attention：keys和values由encoder产生，而queries来自decoder。encoder和decoder在翻译问题中分别接收两种不同的语言，而在文本-图片转化问题中，可以分别接收图片和文字。因此，不同的模态都可以通过Attention连接起来，使得Attention的用途十分广泛。

```python
import torch 
from torch import nn

batch_size = 32
seq_len = 6
n_embd = 10
x = torch.randn(batch_size, seq_len, n_embd)



class MultiHeadAttention(nn.Module):
    '''Implement multiple heads of self-attetition in parallel.'''
    def __init__(self, num_heads, single_head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(single_head_size) for _ in range(num_heads)])


    def forward(self, x):

        return torch.cat([h(x) for h in self.heads], dim=-1)


head_size = 8  
num_heads = 4
single_head_size = head_size // num_heads
multihead_attn = MultiHeadAttention(num_heads, single_head_size)        
print(multihead_attn(x).shape)       # [batch_size, max_len, single_head_size*num_heads]
```

### 总结

3句话：

- Attention的核心是将Query-Key-Value结构用神经网络表示出来（涉及到如何构造这3个部分，以及如何计算权重等，个人理解）。

- Self-Attention更多的是一个特征提取器，通过考虑原始特征之间的关系来构造出更好的特征表示。

- Attention（也称为Cross Attention）可以连接两种不同的模态，从而可以处理很多问题。

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

可能有人会说，Batch Norm好理解，就是把一批数据的每一个维度的样本给标准化一下，可是Layer Norm把一个样本的所有维度给标准化，这种做法似乎没什么道理，比如一个人的身高、体重、年龄，这三个特征的量纲都不同，有啥好标准化的。但其实这个问题在图像和文本应用中就不是一个问题了，因为对于图像来说，某一个位置的特征是不同通道对应的像素值，对于文本来说，一个token的特征是它的embedding向量，可以认为特征之间的量纲都是一样的，也就可以做标准化。

<img title="" src="https://pic.imgdb.cn/item/6401fc17f144a01007fbaa76.jpg" alt="" width="357" data-align="center">

而且在NLP问题中，Layer Norm反而比Batch Norm应用更广泛。~~因为从上图上来看，Batch Norm是按照每句话的位置将数据进行标准化，但是语言是很复杂的，不同位置的词的差异性是很大的，标准化反而可能会损害模型的表现；而Layer Norm是将每一句话进行标准化，相对来说更合理。~~我本来是这么想的，但是后面发现有点问题。

我们假设数据的维度是：[Batch_size (B), seq_len (T), n_embd ($d_e$)]。<font color="red">Transformer中的LayerNorm是对于每个单词的embedding做作标准化。</font>也就是说它计算的均值和标准差的维度是 [B, T, 1]。而我一开始以为的是对于一句话的所有token作标准化，即均值和标准差的维度为 [B, 1, $d_e$]。但其实如果我们紧扣定义：对feature标准化，而token的feature就是它的embedding。

现在已经有许多各种各样的Normalization，感兴趣的可以看看这个博客 [深度学习中的Normalization方法](https://www.cnblogs.com/LXP-Never/p/11566064.html)。

另外还要一点要提一下，就是在原论文中是在每个子模块输出之后（Residual Connection之后）再进行LayerNorm的（post-LN），但是有一些研究发现，对输入做LayerNorm（pre-LN）效果会更好。现在大多是是采用pre-LN，两者的对比见下图。

![](https://pic.imgdb.cn/item/6401fc18f144a01007fbaae0.png)

```python
import torch 
from torch import nn

batch_size = 32
seq_len = 6
n_embd = 10
x = torch.randn(batch_size, seq_len, n_embd)


class LayerNorm(nn.Module):
    "Implements LayerNorm in Transformer. For convenience, just use nn.LayerNorm()."
    def __init__(self, in_dim, eps=1e-6):
        super().__init__()

        self.a = nn.Parameter(torch.ones(in_dim))   # the 'n_embd' dimension
        self.b = nn.Parameter(torch.ones(in_dim))
        self.eps = eps

    def forward(self, x):
        # normalize over the 'n_embd' dimension.
        means = x.mean(dim=-1, keep_dim=True)
        stds = x.std(dim=-1, keep_dim=True)

        out = self.a * (x - means) / (stds + self.eps) + self.b

        return out


layer_norm = nn.LayerNorm(n_embd)
out = layer_norm(x)
print(out.shape)  
```

#### Position-wise Feed-Forward Network

在Encoder中，经过了Self-Attention之后，在输出之前还要在接一个全连接层，使用Relu激活函数，直接上图

<img src="https://pic.imgdb.cn/item/63f84168f144a01007e5dee0.png" title="" alt="" data-align="center">

值得注意的是，对于一句话中的每一个位置是单独使用了一个FFN的，所以称为Positional-wise。但其实每个MLP的权重都是一样的，所以本质上还是只使用了一个FFN。借用知乎上的一张图：

![](https://pic.imgdb.cn/item/63f84168f144a01007e5def2.jpg)

```python
import torch 
from torch import nn

class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),  
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, in_dim),
                                 )


    def forward(self, x):

        return self.net(x)
```

至此，$N \times$框中的内容就介绍完了，接着我们往下看，会发现一个叫Positional Encoding的东西，为啥会有这个呢？这其实是由于Attention机制本身并没有记录与位置相关的信息导致的。

在RNN里，$t$时刻的输入是带有一部分前$t-1$时刻的信息的，因此它天然就是有顺序的。但是Attention不是这样，回顾一下之前的计算，我们发现每一个位置都是与其他所有位置计算权重再加权平均，它们是可以并行计算的，而且打乱词的顺序，得到对应的输出也是一样（这里说的更加严谨一点应该是，打乱key-value的顺序，用同样的$q$得到的$z$是一样的）。因此，Attention不具有位置概念，任意两个词之间的距离都是相等的。那在NLP中，语序毫无疑问是一个重要的特征，因此得想办法把这部分信息引入。

如何引入位置信息呢，做法很直接，我们可以创建一个和输入等长的向量序列，每个向量的维数也和输入的维数相同，这些向量包含位置信息，然后和输入序列相加，从而得到包含位置信息的输入。方法有两种，第一种是**Positional Embedding**。比如，还是以“She is eating a green apple.”为例，假设每个词对应的embedding（在之后会介绍）是$d_e$维，即我们有$(x_1,\ldots,x_6)$，其中$x_i \in \mathbb{R}^{d_e}$。那么我们创建位置向量序列$(v_1,\ldots,v_6)$且$v_i \in \mathbb{R}^{d_e}$。然后通过模型来学习出这些向量到底是什么。但这种方法有一个问题，就是模型在预测阶段只能预测在训练是见到过的位置向量，针对在这个例子，通俗点讲就是一旦碰到句子长度超过6的，模型就不work了。

另一种就是接下来要介绍的Positional Encoding。

#### Positional Encoding

Encoding不进行训练，而是用一个给定的函数来编码位置信息，然后希望模型能通过学习掌握到使用这些信息的方法。这种方法就不会出现上面提到的问题，因为对于任意位置，encoding都能给出一个向量，这样模型至少是能正常进行预测的。Transformer使用的是这种方法，论文中使用的函数如下图所示：

<img src="https://pic.imgdb.cn/item/63f84168f144a01007e5df0f.png" title="" alt="" data-align="center">

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

```python
import numpy as np
import torch 
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=500):
        super(PositionalEncoding, self).__init__()  

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(np.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, embedding):
        "Output token embedding + positional encoding."
        # embedding shape: [batch_size, len, n_embd]
        x = embedding + self.pe[:, :embedding.size(1), :].requires_grad_(False)

        return x
```

#### Input Embedding

最后我们来介绍一下Input Embedding。对于NLP问题，我们最原始的数据是很多句话。我们首先要做的是Tokenization，中文翻译叫做分词，不过对于英文来说，有时候也不一定是根据单词来进行划分，也有可能是根据字符来划分，还有可能是通过sub-word来划分，分出来的一个个部分我们称其为token。这一块具体怎么用我也不太清楚，感兴趣的可以看看这篇博客 [NLP中的Tokenization](https://zhuanlan.zhihu.com/p/444774532)。我们这里就按照分词的概念来做，还是拿“She is eating a green apple.”为例，按词来Tokenize得到的是【She, is, eating, a, green, apple, `<eos>`】。`<eos>`是指end of sequence，也算一个token。计算机是无法直接读取这些token的，所以我们需要对每个token进行编码，这个过程就称为Embedding。再进一步解释，就是我们要为每一个token找到一个embedding向量，使其能够表示这个token。我们拿电影来类别，比如我们要描述一部电影，我们可以从几个维度去描述，比如它属于动作片的部分占多少，属于爱情片的部分占多少，包含喜剧的成分有多少。这些维度是我们自己拍脑袋想出来，对于NLP中的embedding，我们先确定向量的维数，然后通过模型学习出这些embedding vector是什么。当然也可以考虑用其他人训练好的embeddings，Transformer论文中使用的前者。

```python
import torch 
from torch import nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, n_embd):l
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x):

        return self.embedding(x)
```

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

```python
torch.manual_seed(123)
B, T, C = 4, 8, 32  # batch_size, seq_len, n_embd
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)  # [B, T, C] @ [B, C, T] ---> [B, T, T]

# implement sequence mask
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

print(out.shape)   # [4, 8, 16]
```

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

```python
 import torch
 from torch import nn

 class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.smoothing = smoothing

    def forward(self, target):
        # target shape: [batch_size, seq_len, ] or [N, ]
        onehot = F.one_hot(target, self.vocab_size)

        return (1 - self.smoothing) * onehot + (self.smoothing / (self.vocab_size - 1)) * (1 - onehot)


vocab_size = 5
label_smooth = LabelSmoothing(vocab_size)
# target = torch.randint(0, 4, (3, 6))  # [batch_size, seq_len]
target = torch.LongTensor([2, 1, 0, 3, 3])

print(label_smooth(target))        
```

### Implementation

完整的代码我放在这个github仓库里了：[An-Implementation-of-Transformer](https://github.com/leyuanheart/An-Implementation-of-Transformer) 

## GPT （Update at 2023.3.3）

这一部分我们来介绍GPT。它的全称是Generative Pretrained Transformer。从名字上可以看出GPT是一个生成式模型，它通常是用来生成和target风格类似的内容。比如你有一个莎士比亚戏剧的文本集，你希望通过训练使得GPT能够生成带有莎士比亚风格的内容。

和上面提到的翻译问题不同，GPT里使用的Transformer是只有Decoder的部分。而翻译问题（比如德语译英语）是既有target context（英语集），还有一个source context（德语集）所以它需要Encoder来编码source的信息，然后在Decoder中使用。从建模的角度来看的话，在翻译问题中Transformer是在构建条件概率模型$P(Y|X)$，而GPT做的是构建构建边际概率模型$P(Y)$。

**总结一下：在GPT中只使用Transformer的Decoder部分，由于不需要考虑Encoder的信息，所以原来Decoder中的Cross Attention的部分也可以去掉，只保留Self-Attention和FFN即可。另外生成文本的过程也略有不同，主要体现在得到logit之后的操作：翻译问题是通过$\arg \max$来得到下一个token，而GPT是通过将logit传入一个多项分布中，再通过采样来得到下一个token。前者更在乎准确性，而后者是为了实现从目标分布中生成样本。**

### Implementation

GPT的部分我准备以代码为主的形式来写，原理都在之前的介绍的Transformer中了，所以主要还是看如何来应用。以下内容参考Karpathy的[nanoGPT](https://github.com/karpathy/nanoGPT)。

#### Data Preparation

我一直觉得NLP的数据处理也是一个比较复杂的过程。正好也借GPT来完整地过一遍。我们使用的是tiny Shakespeare，一个包含莎士比亚部分文学作品的数据集。大致内容如下：

> First Citizen:
> Before we proceed any further, hear me speak.
> 
> All:
> Speak, speak.
> 
> First Citizen:
> You are all resolved rather to die than to famish?
> 
> All:
> Resolved. resolved.
> 
> First Citizen:
> First, you know Caius Marcius is chief enemy to the people.
> 
> All:
> We know't, we know't.
> 
> First Citizen:
> Let us kill him, and we'll have corn at our own price.
> Is't a verdict?

我们首先要做的是tokenization。对于英文来说，最直接的方法就是按照字符（character）来分。

```python
# get all the unique characters
chars = sorted(list(set(data)))     # chars[0]: '\n',  chars[1]: ' '
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size}")

'''
all the unique characters: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab size: 65
''''


# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(string):
    '''encoder: take a string, output a list of integers'''
    return [stoi[s] for s in string]

def decode(ids):
    '''decoder: take a list of integers, output a string'''
    return ''.join(itos[i] for i in ids)


print(encode("hi there"))
print(decode(encode("hi there")))

'''
[46, 47, 1, 58, 46, 43, 56, 43]
hi there
'''
```

上面使用的是character-level的tokenizer。如果遇到更复杂的问题，可能需要用到一些更好的tokenizers，比如[tiktoken](https://github.com/openai/tiktoken)，[SentencePiece](https://github.com/google/sentencepiece)，[spaCy](https://github.com/explosion/spaCy)。

有了数据，然后就是构造训练集和验证集了（这里不考虑测试集，因为只是为了进行训练的展示。考虑的话也就是多一步的事）。

```python
# create the train, validation split
n = int(0.9 * len(data))   # number of tokens
train_data = torch.as_tensor(encode(data[:n]), dtype=torch.long)
val_data = torch.as_tensor(encode(data[n:]), dtype=torch.long)
print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

'''
train has 1,003,854 tokens
val has 111,540 tokens
'''
```

接下来是最关键的步骤了：明确什么是$x$，什么是$y$。GPT的生成方式是token by token。因此对于模型来说，在时刻$t$，$t$之前的所有tokens都可以是它的输入，而输出就是要预测时刻$t$的token。

我们首先设置一个参数：`context_len`，表示要预测当前时刻的token最多需要之前多少时刻的tokens。以下面的代码为例。对于这样一对长度为8的$x$和$y$，GPT输入$x$，输出的是一个和$y$等长的序列，这个过程中模型一共进行了8步：

1. 输入[18]，输出47对应的预测值

2. 输入[18, 47]，输出56对应的预测值

3. 输入[18, 47, 56]，输出57对应的预测值

4. ...

```python
context_len = 8
x = train_data[:context_len]
y = train_data[1:context_len+1]
print(f"x: {x}")
print(f"y: {y}")
for t in range(context_len):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, the target is {target}.")


'''
x: tensor([18, 47, 56, 57, 58,  1, 15, 47])
y: tensor([47, 56, 57, 58,  1, 15, 47, 58])
when input is tensor([18]), the target is 47.
when input is tensor([18, 47]), the target is 56.
when input is tensor([18, 47, 56]), the target is 57.
when input is tensor([18, 47, 56, 57]), the target is 58.
when input is tensor([18, 47, 56, 57, 58]), the target is 1.
when input is tensor([18, 47, 56, 57, 58,  1]), the target is 15.
when input is tensor([18, 47, 56, 57, 58,  1, 15]), the target is 47.
when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]), the target is 58.
'''
```

上面是一个样本的结果，我们在训练时通常都使用一个batch的样本，所以再构造一个

```python
# create data loader
context_len = 8 
batch_size = 4


def get_batch(data, context_len, atch_size):
    '''generate a batch of data of inputs x and targets y'''
    idx = torch.randint(len(data) - context_len, (batch_size, ))

    x = torch.stack([data[i:i+context_len] for i in idx])
    y = torch.stack([data[i+1:i+context_len+1] for i in idx])

    return x, y



xb, yb = get_batch(train_data, context_len, batch_size)
print(xb.shape)  # [4, 8]
print(yb.shape)  # [4, 8]
```

至此，数据准备就完成了，接下来就是构造GPT了。

#### Build GPT

这一部分里的内容在前面Transformer里都有介绍，而且由于只涉及到了Decoder，所以所有的Attention都是Self-Attention，好些代码还可以简化。另外有一点需要提一下，之前Transformer里用到了Positional Encoding，我提到了其实也可以和token embedding一样学一个positional embedding，所以这里就是用了后者，也算是两种方法都体验一下。我就直接放代码了。

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, context_len, d_model, h, dropout=0.1):
        super().__init__()

        assert d_model % h == 0
        self.h = h
        self.d = d_model // h

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(p=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.attn = None
        self.register_buffer('mask', torch.tril(torch.ones(1, 1, context_len, context_len)).type(torch.int8))

    def forward(self, x):
        # x shape: [batch_size, len, d_model]
        # mask shape: [1, 1, len, len]
        b, t, d_model = x.shape  

        # pre-attention linear projection:         
        q, k, v = self.qkv(x).split(d_model, dim=-1)

        # separate different heads: [b, len, h*d] ==> [b, len, h, d]
        q = q.view(b, t, self.h, self.d)
        k = k.view(b, t, self.h, self.d)
        v = v.view(b, t, self.h, self.d)

        # transpose for attention dot product: [b, len, h, d] ==> [b, h, len, d]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        att = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.attn = att
        att = self.attn_dropout(att)   # Not mentioned in the paper but practically useful

        out = att @ v            # [b, h, len, d]
        # transpose to move the head dimension back: [b, h, len, d] ==> [b, len, h, d]
        # combine the last two dimensions to concatenate all the heads together: [b, len, h*d]
        out = out.transpose(1, 2).contiguous().view(b, t, d_model)
        out = self.proj(out)  # [b, h, len, d]

        del q, k ,v

        return self.dropout(out)


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.w1 = nn.Linear(d_model, 4 * d_model)  
        self.w2 = nn.Linear(4 * d_model, d_model)  
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: [b, len, d_model]
        x = self.w2(F.relu(self.w1(x)))         

        return self.dropout(x)



class DecoderBlock(nn.Module):
    def __init__(self, context_len, d_model, h, dropout):
        super(DecoderBlock, self).__init__()
        self.dec_attn = MultiHeadSelfAttention(context_len, d_model, h, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # pre-LayerNorm
        x = x + self.dec_attn(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


class Decoder(nn.Module):
    def __init__(self, context_len, d_model, h, N, dropout):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(*[DecoderBlock(context_len, d_model, h, dropout) for _ in range(N)])  

        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, seq):

        dec_output = self.decoder(seq)

        return self.layer_norm(dec_output)




class GPT(nn.Module):
    def __init__(self, vocab_size, context_len, d_model, h, N, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(context_len, d_model)   # Instead of using encoding as the original paper, embedding is used here
        self.decoder = Decoder(context_len, d_model, h, N, dropout)
        self.linear = nn.Linear(d_model, vocab_size)

        self.context_len = context_len

        # # initialize the weights    
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, seq):
        # seq shape: [batch_size, seq_len]
        b, t = seq.shape
        token_emb = self.token_embedding(seq)  # [b, t, d_model]
        pos_emb =self.positional_embedding(torch.arange(t)) # [t, d_model]
        x = token_emb + pos_emb   # [b, t, d_model]

        x = self.decoder(x)  # [b, t, d_model]

        logits = self.linear(x)   # [b, t, vocab_size]

        return logits



    @torch.no_grad()
    def generate(self, seq, max_len):
        # seq is [B, t] array of indices in the current context
        for _ in range(max_len):
            # crop seq to the last 'context_len' tokens, or the positional embedding will fail
            seq_crop = seq[:, -self.context_len:]
            logits = self(seq_crop)
            # focus on only the last time step
            logits = logits[:, -1, :]   # [B, vocab_size]
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            # append sampled index to the running sequence
            seq = torch.cat([seq, idx_next], dim=-1)  # [B, t+1]

        return seq
```

#### Training

训练部分主要就是构造损失、设置优化器、写迭代的过程、记录一些中间结果等，我也直接放代码了，然后给一段经过训练的GPT生成的文本。

```python
def compute_loss(model, x, y):
    # x, y shape:[b, t]
    logits = model(x)
    b, t, vocab_size = logits.shape
    logits = logits.contiguous().view(-1, vocab_size)
    y = y.contiguous().view(-1)
    loss = F.cross_entropy(logits, y)

    return loss


@torch.no_grad()
def eval_loss(model, eval_iters):
    train_loss = val_loss = 0

    model.eval()
    for _ in range(eval_iters):
        x, y = get_batch(train_data, context_len, batch_size)
        train_loss += compute_loss(model, x, y).item()

        x, y = get_batch(val_data, context_len, batch_size)
        val_loss += compute_loss(model, x, y).item()        
    model.train()

    return train_loss / eval_iters, val_loss / eval_iters


vocab_size = len(chars)
context_len = 8  # context of up to 8 previous characters
d_model = 32     # token embedding dim
h = 4            # number of heads in self-attention   
N = 2            # number of blocks in decoder
dropout = 0.1
batch_size = 32
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500  # keep frequent because we'll overfit
eval_iters = 200  

seed = 3
torch.manual_seed(seed)

gpt = GPT(vocab_size, context_len, d_model, h, N, dropout)
optimizer = optim.Adam(gpt.parameters(), lr=learning_rate)


train_losses = []
val_losses = []
start = time.time()
for step in range(max_iters):   
    # every once in a while evaluate the train and val loss
    if step % eval_interval == 0:
        train_loss, val_loss = eval_loss(gpt, eval_iters)
        # print('='*30)
        print(f"step: {step}, train loss: {train_loss:.4f}, eval loss: {val_loss:.4f}, time: {timedelta(seconds=time.time()-start)}")
        # print('='*30)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    xb, yb = get_batch(train_data, context_len, batch_size)

    loss = compute_loss(gpt, xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# plt.plot(train_losses, label='train_loss')
# plt.plot(val_losses, label='val_loss')
# plt.legend()


gpt.eval()
start = torch.zeros((1, 1), dtype=torch.long)
print(decode(gpt.generate(start, max_len=500)[0].tolist()))


'''
May, ber: sahno to wlearis,.

CENCES: say!

QUY I meses!,
Ny hey,
'King be digs it nebone, daceelt you hord weake?
Old in ais Leak-Fuaty nop that muse so-mest:
I'll for yenest you me his a king and yuuncair gase coush liene, is whyfelly hou eak of to not in denciold
pot old an Bate one adtle:
Nowees khe they hond.

KING me stevise, this Garr, Nodsinnashing to od be a with your, grienim us um theed and the reard hes; mary stelt, my.

NRY med an that not Litel, in have, be shees by no's eater.
Tin
'''
```

可以看到已经有点那么回事了，虽然许多单词还是错的，但是结构上已经有那种一个人说一段话的意思了。想要更好表现的话就得把模型的规模扩大、训练更长时间了。

GPT这个部分的代码我也放在这个仓库了：[An-Implementation-of-Transformer](https://github.com/leyuanheart/An-Implementation-of-Transformer)

终于写完啦！！！

## 参考资料

[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[Self-Attention和Transformer](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#skip-connection-he-layer-normalization)

[Transformer10个重要问题](https://mp.weixin.qq.com/s/OSTJx-JSpoeMVEKZRjVZXA)

[Transformer 知识点理解](https://zhuanlan.zhihu.com/p/58009338)

[[综述] A survey of Transformers-[7] LayerNorm和FFN]( https://zhuanlan.zhihu.com/p/460854500)

[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/#attention-visualization)

[nanoGPT代码](https://github.com/karpathy/nanoGPT)

[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) （如果不能看YouTube，B站上也有）
