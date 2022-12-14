---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter2
subtitle:   Multi-armed Bandits
date:       2021-04-03
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />


# 多臂赌博机（Multi-armed Bandits）

$\epsilon$贪心方法相对于贪心方法的优点依赖于任务。比方说，如果reward的方差很大，那么为了找到最优的动作需要多次的试探。但是，如果方差为0，那么贪心方法会在一次尝试之后就知道每个动作的真实价值。在这种情况下，贪心方法实际上可能表现最好。

然而，如果任务是非平稳的，即动作的真实价值会随着时间而变化。那么，即使在确定性的情况下，试探也是需要的，这是为了确认某个非贪心动作不会变得比贪心动作更好。而**非平稳是强化学习中最常遇到的问题**。


$$
\text{新估计值} \longleftarrow \text{旧估计值} + \text{步长} \times [\text{目标} - \text{旧估计值}]
$$


表达式$[\text{目标} - \text{旧估计值}]$是估计值的**误差**。误差会随着向“目标”（Target）靠近的每一步而减小。虽然“目标”中可能充满噪声，但我们还是假定“目标”会告诉我们可行的前进方向。

![1](https://pic.imgdb.cn/item/60fbb0085132923bf858f535.png)

步长满足上图条件可以保证序列收敛概率为1。不过符合上图的步长参数序列常常收敛得很慢，或者需要大量的调试才能得到一个满意的收敛率。尽管在理论工作中很常用，但在实际应用中很少用到。

## 乐观初始值（Optimistic Initial Values）

![2](https://pic.imgdb.cn/item/60fbb0085132923bf858f543.png)

![3](https://pic.imgdb.cn/item/60fbb0085132923bf858f551.png)

这种更新在一定程度上依赖于初始动作值$Q_1(a)$的选择。从统计学的角度来说，这些方法是**有偏**的。对于采样平均法来说，当所有动作都至少被选择一次时，偏差就会消失。但是对于步长为常数$\alpha$的情况，偏差会逐渐减小，但不会消失。在实际问题中，这通常不是一个问题。

缺点是：如果不将初始值全部设为0，则初始值实际上变成了一个必须由用户选择的参数集。

好处是：通过它们可以简单地设置关于预期收益水平的**先验知识**。

初始动作的价值也提供了一种简单的试探方式，就是给出**乐观**的初始估计。因为无论哪种动作被选择，收益都比最开始的估计要小，因此学习器会对得到的收益感到“失望”，从而转向另一个动作。这个技巧在平稳问题中非常有效，但是不太适合非平稳问题，因为它试探的驱动力天生是暂时的。

### 无偏恒定步长技巧

我们是否有办法既能利用恒定步长在非平稳过程中的优势，又能有效避免它的偏差呢？

一种可行的方法是利用如下的步长来处理某个特定动作的第$n$个收益

![4](https://pic.imgdb.cn/item/60fbb0085132923bf858f567.png)

其中，$\alpha>0$是一个传统的恒定步长，$\hat{o}_n$是一个从零时刻开始计算的修正系数

![5](https://pic.imgdb.cn/item/60fbb0085132923bf858f57a.png)

可以证明$Q_n$是一个无初始值偏差的指数近因加权。

## 基于置信上界（Upper-Confidence-Bound）的动作选择

因为对动作-价值的估计总会存在不确定性，所以试探是必须的。$\epsilon$贪心算法会尝试选择非贪心的动作，但是这是一种**盲目**（完全随机）的选择。它不会对接近贪心或者不确定性特别大的动作有偏好。而在这些非贪心动作中，如果能根据它们的潜力来选择可能事实上是更好的策略。一个有效的方法就是按照下面的公式来选择动作

![6](https://pic.imgdb.cn/item/60fbb0765132923bf85acbfb.png)

$N_t(a)$表示在时刻$t$之前动作$a$被选择的次数。$c$是一个大于0的数，它控制试探的程度。如果$N_(t)=0$，则$a$就被认为是满足最大化条件的动作。

这种基于**置信度上界**（upper confidence bound, UCB）的动作选择的思想是，平方根项是对动作$a$值估计的不确定性或方差的度量。因此，最大值的大小是动作$a$的可能真实值的上限，参数$c$决定了置信水平。每次选择$a$，$N_t(a)$增大，这一项就减小了。另一方面，每次选择动作$a$之外的动作时，由于$t$的增大，不确定性又增加了。

和$\epsilon$贪心算法相比，它更难推广到一些更一般的强化学习问题。尤其是要处理大的状态空间，涉及到函数近似问题，目前还没有已知的使用方法利用UCB动作选择的思想。

## 梯度赌博机算法（Gradient Bandit Algorithms）

我们也可以不使用动作-价值，而是针对每个动作$a$考虑学习一个数值化的**偏好**函数$H_t(a)$，偏好函数越大，动作就越频繁地被选择，但是偏好函数的概念并不是从“收益”的意义上提出来的。**只有一个动作相对于另一个动作的相对偏好才是重要的**。有了偏好函数，我们就可以按照softmax分布（吉布斯或玻尔兹曼分布）确定每个动作被选择的概率。

![7](https://pic.imgdb.cn/item/60fbb0765132923bf85acc11.png)

基于梯度上升的思想，在每个步骤中，在选择动作$A_t$并获得收益$R_t$之后，偏好函数将会按照如下方式更新

![8](https://pic.imgdb.cn/item/60fbb0765132923bf85acc28.png)

$\hat{R}_t$是在时刻$t$之前所有收益的平均值，作为比较收益的一个基准项（baseline）。如果收益高于它，那么在未来选择动作$A_t$的概率就会增加，反之概率就会降低，未选择的动作被选择的概率上升。

![9](https://pic.imgdb.cn/item/60fbb0765132923bf85acc44.png)

注意，对于收益基准项，除了要求它不依赖于所选的动作之外，不需要其他任何的假设。基准项的选择不影响算法的**期望**更新值，但是它确实会影响更新值的方差，从而影响收敛速度。选择收益的平均值作为基准项可能不是最好的，但它很简单，并且在实践中很有效。

## 关联搜索（Associative Search）

目前为止，只考虑了非关联的任务，对它们来说，没有必要将不同的动作与不同的情境联系起来。

当任务是平稳的，学习器会试图寻找一个最优的**动作**；当任务是非平稳的，最佳动作会随着时间的变化而变化，此时它会试着去追踪最佳动作。

然而在一般的强化学习任务中，往往有不止一种情境，它们的目标是学习一种**策略**：一个从特定情境到最优动作的映射。

举个例子，假设有一系列不同的$k$臂赌博机任务，每一步你都要随机地面对其中的一个。从观察者的角度来看。这是一个单一的、非平稳的$k$臂赌博机任务，可以尝试使用常数步长的更新等方法来处理这个非平稳情况。但是，除非真正的动作-价值的改变是非常缓慢的，否则这些方法不会有好的结果。

现在假设，当你遇到某一个$k$臂赌博机任务时，你会得到关于这个任务的编号的明显线索（但不是它的动作-价值）。比如不同的臂赌博机的颜色不同。那么，现在你可以学习一些任务相关的操作**策略**，例如，用颜色作为信号，把每个任务和该任务下的最优动作直接关联起来，比如，如果为红色，则选择1号臂；如果为绿色，则选择2号臂。有了这种任务相关的策略，在知道任务编号信息时，通常要比不知道时做得更好。

这就是关联搜索任务，文献中通常也称为“上下文相关的老虎机”（contextual bandits）。它既涉及到采用试错学习去搜素最优动作，又将这些动作与它们表现最优时的情境关联在一起。

关联搜索任务介于$k$臂赌博机问题和完整强化学习问题之间。它与完整强化学习问题的相似点是，它需要学习一种**策略**。但它又与$k$臂赌博机问题相似，体现在每个动作只影响即时收益。如果允许动作可以影响下一时刻的情境和收益，那么这就是完整的强化学习问题了。