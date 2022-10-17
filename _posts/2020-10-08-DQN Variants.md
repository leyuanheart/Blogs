---
layout:     post
title:      Reinforcement Learning 5
subtitle:   QDN Variants
date:       2020-10-08
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---


这一篇我们将一些DQN的变化版本，为什么会出现这些版本呢，首先就要说到DQN的一个问题：**过高估计Q value**。

## Overestimation of Q value

我们回顾一下Q-learning的target值：$y_t=r_t+\gamma \underset{a'}{\max}Q_w(s_{t+1},a')$

这个`max`操作会使Q value越来越大，甚至高于真实值，原因用数学公式来解释的话就是：对于两个随机变量$X_1$和$X_2$，有$E[\max(X_1,X_2)]\geq \max(E[X_1],E[X_2])$


$$
\begin{align}
\underset{a'}{\max}Q_w(s_{t+1},a') & = Q_w(s_{t+1}, \arg \underset{a'}{\max}Q_w(s_{t+1},a')) \\
& =E[G_t|s_{t+1}, \arg \underset{a'}{\max}Q_w(s_{t+1},a'),w] \\
& \geq max(E[G_t|s_{t+1},a_1,w],E[G_t|s_{t+1},a_2,w],\cdots) \\
& = max(Q_w(s_{t+1},a_1), Q_w(s_{t+1},a_2),\cdots)
\end{align}
$$


不等式右边才是我们真正想求的值，问题就出在选择action的网络和计算target Q value的网络是同一个网络。而且随着候选动作的增加，过高估计的程度也会越严重（可以理解成动作集的增加会增加随机性，更容易产生一个过高的估计）。

## Double DQN 

[Deep Reinforcement Learning with Double Q-Learning. Van Hasselt et al, AAAI 2016](https://arxiv.org/pdf/1509.06461.pdf)

论文里展示了一下过高估计的情况

![32](https://pic.downk.cc/item/5f7eb2b91cd1bbb86bab701e.png)

![33](https://pic.downk.cc/item/5f7eb2b91cd1bbb86bab7023.png)

Double DQN的想法就是用两个`network`来分别做动作的选择和计算target，使得估计更稳健，而且这个想法在应用的时候也非常方便，因为本来DQN中就有两个network：一个target model，一个training model。

传统的DQN (Vanilla DQN)的target:


$$
y_t=r_t+\gamma Q_{w^-}(s_{t+1}, \arg \underset{a'}{\max}Q_{w^-}(s_{t+1},a'))=r_t+\gamma \underset{a'}{\max}Q_{w^-}(s_{t+1},a')
$$


Double DQN的target：


$$
y_t=r_t+\gamma Q_{w^-}(s_{t+1}, \arg \underset{a'}{\max}Q_w(s_{t+1},a'))
$$


结果确实能降低Q value的过高估计程度。

![34](https://pic.downk.cc/item/5f7eb2b91cd1bbb86bab7029.png)

## Dueling DQN 

[Dueling Network Architectures for Deep Reinforcement Learning. Wang et al, best paper ICML 2016](https://arxiv.org/pdf/1511.06581.pdf)

Dueling DQN是从改变网络架构上来入手的，下面是它的网络设计图：

![35](https://pic.downk.cc/item/5f7eb2b91cd1bbb86bab7033.png)

它将原来对Q value的估计拆成了2部分：一部分是$V(s)$，一部分是$A(s,a)$，叫做**Adavantage function**。

我们可以分析一下为什么要这么做，首先我们知道$V(s)=E[Q(s,a)]$，那么$Q(s,a)=V(s)+A(s,a)$想表达的意思就是，将$Q(s,a)$拆解成对状态$s$的一个估计和在状态$s$的基础上对不同action的一个衡量指标。这是挺符合实际的一种拆分法，拿游戏来打比方，有些状态它本身就是很好的，这时你其实随便采取什么动作都是OK的，但是有些状态可能比较困难，这时不同动作之间的差异就很大了。

另外还有一个好处就是可以更高效地学习$Q(s,a)$，以下面的图来说明：

![36](https://pic.downk.cc/item/5f7eb2b91cd1bbb86bab703e.png)

竖着的是action (3个)，横着的是state(4个），假如现在我们想让Q里的3、-1变成4、0，但是我们不是直接去更新Q，而是更新V和A。如果是V中的0变成1，那么我们会发现，不只是3和-1变化了，-2也跟着变化了，这样做其实是更有效率的，因为我们不用去采集到动作3，也能对它的值进行更新。

但这只是我们想要这个网络达到的效果，但是实际上也可能会出现训练出来V=0，然后Q=A。所以必须对网络施加限制，原论文中也很多种类的限制，比如把A的列减去它的均值或者最大值，然后再和V相加构成Q。

## Prioritized Experience Replay

[Prioritized Experience Replay. Schaul et al, ICLR 2016](https://arxiv.org/pdf/1511.05952.pdf)

这个方法的思想是：并不是每一个样本都是同样重要的，而是那些算出来TD error比较大的样本需要更多地被关注，因为error大意味着我们没有学好，还有很多可以学习的地方。

它的做法就是对于在replay memory里的transition tuple加一个权重，表示每个样本的学习价值，每次按照这个权重来抽样，具体操作如下：

对于replay memory里的抽样：

对于每一个transition tuple，计算TD error：$\delta_i=r_t+\gamma \underset{a'}{\max}Q_w(s_{t+1},a')-Q_w(s_t,a_t)$

然后对于每个样本的抽样概率：$P(i)=\frac{p_i^{\alpha}}{\sum_kp_k^{\alpha}}$

其中$p_i$表示transition $i$的priority，这个priority可以不同的设定，比如$p_i=\|\delta_i\|+\epsilon$，$\epsilon$是一个很小的正数，为了防止出现如果error为0，那么这个样本就不会被采样到的情况发生；另一种设定基于error排序，$p_i=\frac{1}{rank(i)}$，$\alpha$是一个超参数，来确定要分配多少prioritization，$\alpha=0$就意味着等权重抽样。

随机梯度下降是基于它的均值是整体梯度的一个无偏估计，由于抽样方法的改变使抽样数据的分布发生了改变，所以如果还和之前一样更新参数，会造成偏差，所以相应的参数更新也要进行纠偏（通过importance-sampling weights)：$w_i=(\frac{1}{N}\frac{1}{P(i)})^{\beta}$，$\beta$也是一个超参数，$\beta=1$表示完全去纠正这个偏差（实际操作上是先取一个小于1的值，然后随着训练的进行慢慢增加到1），在更新的时候就用$w_i\times\delta_i$，通常我们还会把weights给normalize一下，除以$max_iw_i$。

## Noisy DQN

[Parameter space noise for exploration. OpenAI ICLR 2018](https://arxiv.org/pdf/1706.01905.pdf)

[Noisy networks for exploration. DeepMind ICLR 2018](https://arxiv.org/pdf/1706.10295.pdf)

DeepMind和OpenAI两家强化学习巨头同时在ICLR 2018上发表了思想几乎完全一样的两篇论文，差别基本上就是加的noise不太一样。

一下的一些图片和内容是基于[李宏毅老师强化学习的课程](https://www.bilibili.com/video/BV1UE411G78S?p=7)整理的。

首先回顾一下之前提到过的为了增加探索的成分而使用的$\epsilon$-greedy：


$$
a = \begin{cases}
\arg \underset{a}{\max} Q(s,a), & with\ probability\  1-\epsilon \\
random,   & otherwise
\end{cases}
$$


这是在action上加noise，意思是对于一个给定的state，agent可能会采取不同的action，但是其实这并不是很符合一个策略实际上做决策的过程。现实中对于同一个给定的state，agent应该是要采取相同的action的（这也是我为什么觉得stochastic策略不是那么好应用的原因）。

Noisy DQN的思想就是：在参数上加noise：

![37](https://pic.downk.cc/item/5f7eb3101cd1bbb86bab8630.png)

而且需要注意的是`是在每个episode开始的时候给Q function的参数加noise`，这样就能保证在一次游戏中agent能以一致的行为进行探索 (explore in a consistent way)，文章中叫做`state-dependent exploration`。

## Distributional DQN

[A Distributional Perspective on Reinforcement Learning.  ICML 2017](https://arxiv.org/pdf/1707.06887.pdf)

我们想求的Q其实本质上是一个期望：$Q^{\pi}(s,a)=E_{\pi}[G_t\|S_t=s,A_t=a]$，但是不同的$G_t$分布可能会有相同的期望。

![38](https://pic.downk.cc/item/5f7eb3101cd1bbb86bab8635.png)

那一个自然的想法，如果能把$G_t$的分布给估计出来，是不是能更好地做决策。

做法也比较直接，就是先给定$G_t$的一个range，然后划分成一定数量的bins (文章中叫atoms)。比如本来action的维度是3，输出的3个Q，现在假设设置atoms为5，则要输出15个heads（文章中表现最好的是51个atoms，称作C51）。在做决策的时候还是average每个action的分布得到Q，取最大的Q对应的action。（我觉得也可以用不同的分位数作为选择标准，比如对于一些需要考虑安全性的问题，我们是需要它的最差表现不能太差，这时可以选比较小的分位数）

![39](https://pic.downk.cc/item/5f7eb3101cd1bbb86bab863c.png)

## 代码

这一部分代码很多，我就不放在文章里，如果想要看看代码是怎么写的，我把相应的代码放在我的GitHub仓库[Reinforcement_Learning_Algorithms](https://github.com/leyuanheart/Reinforcement_Learning_Algorithms)上了，里面是我整理的关于RL的代码，还在持续更新中。