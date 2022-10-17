---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter12
subtitle:   Eligibility Traces
date:       2021-07-23
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

资格迹是强化学习的基本方法之一。它将时序差分和蒙特卡洛算法统一了起来并进行了拓展。它也提供了一种方法使得蒙特卡洛算法一方面可以在线使用，另一方面可以在不是分幕式的持续性问题上使用。

直接介绍的n-步时序差分算法是一种统一时序差分和蒙特卡洛算法的方式，但是资格迹在此基础上给出了具有明显计算优势的更优雅的算法机制。这个机制的核心是一个短时记忆向量，资格迹$\mathbf{z}_t \in \mathbb{R}^d$，以及与之对应的长时权重向量$\mathbf{w}_t \in \mathbb{R}^d$。这个方法的核心思想是，当参数$\mathbf{w}_t$的一个分量参与计算并产生一个估计值时，对应的$\mathbf{z}_t$的分量会骤然升高，然后逐渐衰减。在迹归零前，如果发现了非零的时序差分误差，那么相应的$\mathbf{w}_t$的分量就可以得到学习。迹衰减参数$\lambda \in [0,1]$决定了迹的衰减率。

在n-步算法中资格迹的主要计算优势在于，它只需要追踪一个迹向量，而不需存储最近的n个特征向量（这个优点当使用表格型表征时就不那么明显了）。同时，学习也会持续并统一地在整个时间上进行，而不是延迟到整幕的末尾才进行。（这两句话当你看完这篇之后如果能有些理解的话，说明你对资格迹是懂了一点的）

n-步时序差分算法基于接下来的n步的收益以及n步之后的状态来更新。这些通过带更新的状态往前看得到的算法，被称为**前向视图**。前向视图的一般实现比较复杂，因为它的更新依赖于当前还未发生的未来。然而，使用当前的时序差分误差并用资格迹往回看那些已访问过的状态，就能够得到几乎一样（有时甚至完全一模一样）的更新。这种替代的学习算法称为资格迹的**后向视图**。

## $\lambda$-回报

n-步回报的定义

![231](https://pic.imgdb.cn/item/60fbd6ac5132923bf8fb5e5f.png)

我们除了可以用任意的n步回报作为目标外，也可用不同的n的平均n步回作为更新目标。这样的复合型回报和单个的n步回报一样都具有能够减小误差的性质，因此可以保证更新的收敛性。一个复合更新只能在它的组分中最长的那个更新完成后完成。

可以把TD($\lambda$)算法视为平均n步更新的一种特例。这里的平均值包括所有可能的n步更新，每一个按照$\lambda^{n-1}$加权，最后乘以正则项$(1-\lambda)$保证权值和为1. 产生的结果称为回报，定义为

![232](https://pic.imgdb.cn/item/60fbd6ac5132923bf8fb5e6c.png)

在达到一种终止状态后，所有的后续n步回报等于$G_t$。如果愿意的话，可以将这些终止状态之后的计算项从主要的求和项中独立出来，得到

![233](https://pic.imgdb.cn/item/60fbd6ac5132923bf8fb5e83.png)

最后一项$\lambda^{T-t-1}G_t=(1-\lambda)\sum_{n=T-t}^{\infty}\lambda^{n-1}G_t$。

通过这个公式，我们可以很清楚地看到，当$\lambda=1$时，只剩下$G_t$，此时$\lambda$-回报的更新算法就是蒙特卡洛算法。当$\lambda=0$时，$\lambda$-回报为$G_{t:t+1}$,即单步回报，此时，$\lambda$-回报的更新就是单步时序差分算法。

![234](https://pic.imgdb.cn/item/60fbd6ac5132923bf8fb5e99.png)

![235](https://pic.imgdb.cn/item/60fbd6ac5132923bf8fb5eaa.png)

我们现在可以定义基于$\lambda$回报的第一个学习算法了，即离线$\lambda$-回报算法。作为一个离线算法，在一幕序列中间不会改变权值向量，而是在整幕结束后，才会进行整个序列的离线更新。

![236](https://pic.imgdb.cn/item/60fbd6f15132923bf8fc6c95.png)

在19-状态随机游走问题上检测其效果。和n-步算法相当。我们目前采取的所有算法，理论上都是前向的。

![237](https://pic.imgdb.cn/item/60fbd6f15132923bf8fc6cb7.png)

## TD($\lambda$)

TD($\lambda$)是第一个使用资格迹展示了更理论化的前向视图和更易计算的后向视图之间关系的算法。下面将以实证的方式展示它近似于上面提到的离线$\lambda$-回报算法。

TD($\lambda$) improves over the off-line "$\lambda$-return algorithm in three ways. First it updates the weight vector on every step of an episode rather than only at the end, and thus its estimates may be better sooner. Second, its computations are equally distributed in time rather than all at the end of the episode. And third, it can be applied to continuing problems rather than just to episodic problems. In this section we present the semi-gradient version of TD($\lambda$) with function approximation.

资格迹是一个短期记忆，其持续时间通常少于一幕的长度。资格迹辅助整个学习过程，它们唯一的作用是影响权重向量。资格迹向量被初始化为零，然后在每一步累加价值函数的梯度，以$\gamma \lambda$衰减

![238](https://pic.imgdb.cn/item/60fbd6f15132923bf8fc6cd2.png)

资格迹追踪了对最近的状态评估值做出了或正或负贡献的权值向量的分量，这里的“最近”由$\gamma \lambda$来定义。当一个强化事件出现时，我们认为这些贡献“痕迹”展示了权值向量的对应分量有多少“资格”可以接受学习过程中引起的变化。我们关注的强化事件是一个又一个时刻的单步时序差分误差。

![239](https://pic.imgdb.cn/item/60fbd6f15132923bf8fc6ce6.png)

如果$\lambda=0$，在t时刻的迹恰好为$S_t$对应的价值函数的梯度。此时的TD($\lambda$)就退化为单步半梯度时序差分。TD(0)仅仅让当前时刻的前导状态被当前时刻的时序差分误差所改变。对于更大的$\lambda$，更多的之前的状态会被改变，越远的状态改变越少，这是因为资格迹更小。

如果$\lambda=1$，那么之前状态的“信用”每步仅仅衰减$\gamma$。这个恰好与蒙特卡洛算法的行为一致。比如说，对于$R_{t+1}$，在更新$V(S_t)$时是无折扣的使用它的，在更新$V(S_{t-1})$时是用的$\gamma R_{t+1}$，在$V(S_{t-2})$时是用的$\gamma^2 R_{t+1}$。如果$\lambda=\gamma=1$，那么资格迹在任何时刻都不衰减，该算法与无折扣的分幕式任务的蒙特卡洛算法的表现就是完全一样的。

同样在19-状态随机游走的问题中检验效果。对于每一个$\lambda$值，如果选择最优的$\alpha$或者稍微小一点的值，两种算法的性能几乎是相同的。但是如果$\alpha$的值比最优值大很多，$\lambda$-回报仅仅变差一点点，但是TD($\lambda)$则变差了很多，甚至可能不稳定。

![240](https://pic.imgdb.cn/item/60fbd6f15132923bf8fc6cfe.png)

线性TD($\lambda$)被证明会在同轨策略的情况下收敛。这里的收敛并不会正好收敛到对应于最小误差的权值向量，而是根据$\lambda$收敛到它附近的一个权值向量。对于一个带折扣的持续性任务

![241](https://pic.imgdb.cn/item/60fbd71e5132923bf8fd2a69.png)

当$\lambda$接近于1时，这个上界接近最小误差。然而，实际上$\lambda=1$通常是一个最差的选择。

![242](https://pic.imgdb.cn/item/60fbd71e5132923bf8fd2a85.png)

## n-步截断$\lambda$回报方法

离线$\lambda$-回报算法描述的是一种理想状况，但其作用有限，因为它使用了直到幕末尾才知道的$\lambda$-回报。在持续性任务的情况下，严格来说$\lambda$-回报是永远未知的，因为它依赖于任意大的n所对应的n-步回报。然而，由于每一时刻有$\gamma \lambda$的衰减，实际上对长期未来的收益的依赖是越来越弱的。那么一个自然的近似便是截断一定步数之后的序列。

一般的，假定数据最远只能达到未来的某个视界h。

![243](https://pic.imgdb.cn/item/60fbd71e5132923bf8fd2a9f.png)

于是很自然的就有了新的n-步截断$\lambda$-回报算法，被称为截断TD($\lambda$)或TTD($\lambda$)。

![244](https://pic.imgdb.cn/item/60fbd71e5132923bf8fd2ab6.png)

![245](https://pic.imgdb.cn/item/60fbd71e5132923bf8fd2ad7.png)

## 重做更新：在线$\lambda$-回报算法

选择截断TD($\lambda$)中的截断参数n时需要折中考虑。一方面，n应该尽量大使其更接近离线$\lambda$-回报算法，另一方面，它也应该比较小使得更新更快且迅速地影响后续行为。我们是否能找到最好的折中？原则上可以，但是必须以增加计算复杂性为代价。

<font color=red>基本思想是，在每一步收集到新的数据增量的同时，回到当前幕的开始并重做所有的更新。</font>The new updates will be better than the ones you previously made because now they can take into account the time step’s new data. That is, the updates are always towards an n-step truncated $\lambda$-return target, but they always use the latest horizon. In each pass over that episode you can use a slightly longer horizon and obtain slightly better results.

如果计算复杂性不是问题，如何理想地使用n步截断$\lambda$​​​​​​​​-回报。幕开始的时候，在时刻0使用上一幕末尾的权值$\mathbf{w}\_0$​​​​​​​​。在视界扩展到时刻1时开始学习。给定到视界1为止的数据，0时刻的估计目标只能是单步回报$G_{0:1}$​​​​​​​​，也是$G_{0:1}^{\lambda}$​​​​​​​​。用这个目标进行更新，可以得到$\mathbf{w}\_1$​​​​​​​​。然后，随着视界扩展到时刻2，我们有从新数据$S_2,R_2$​​​​​​​​，以及新的权值$\mathbf{w}\_1$​​​​​​​​，所以我们现在可以为第一个$S_0$​​​​​​​​的更新构建一个更好的目标$G_{0:2}^{\lambda}$​​​​​​​​。采用这些改善过的目标，我们从$\mathbf{w}\_0$​​​​​​​​开始重做$S_1$​​​​​​​​和$S_2$​​​​​​​​的更新，就得到了$\mathbf{w}\_2$​​​​​​​​，以此类推。当视界每扩展一步，我们都要从$\mathbf{w}\_0$​​​​​​​​开始重做所有的更新。这个过程要使用在前面视界下得到的权重向量（**在构建新的目标$G_{0:t}^{\lambda}$​​​​​​​​的时候用**）。

这个概念性的算法涉及了对整幕序列的多次遍历，对每个不同的视界都要遍历一次，每次都生成一个不同的权值序列。为了清晰地描述这个算法，我们必须区分不同视界计算的不同权值向量。我们定义$\mathbf{w}_t^h$为在视界h的序列的时刻t所生成的权值。第一个权重向量$\mathbf{w}_0^h$由上一幕继承而来，最后一个权值向量$\mathbf{w}^h$为算法的最终权值向量。最终一个视界$h=T$获得的最终权值$\mathbf{w}_T^T$会被传送给下一幕用作初始权值向量。

![246](https://pic.imgdb.cn/item/60fbd74f5132923bf8fe099f.png)

在线$\lambda$-回报算法的权重的序列可以被组织成一个三角形。在每个时刻产生这个三角形的一行。只有对角线上的权重向量$\mathbf{w}_t^t$是我们真正需要的，在n-步自举法更新回报值中产生作用。在最终算法中对角权重向量被重命名了，取消了上标，$\mathbf{w}_t \doteq \mathbf{w}_t^t$。

在线$\lambda$​-回报算法是一个完全在线的算法。其在每一幕中的每一个时刻t，仅仅使用时刻t获取的信息确定新的权值向量$\mathbf{w}\_t$​。它的主要缺点是计算非常复杂，每一个时刻需要遍历整幕序列。它比离线$\lambda$​-回报算法要复杂，因为后者在终止时虽然也需要遍历整幕，但是在该幕中间不会做任何的更新。In return, the online algorithm can be expected to perform better than the off-line one。这不仅仅是因为在幕中见进行更新，而且也因为在幕结束时，用于自举法的权值向量（在$G_{t:h}^{\lambda}$​）获得了更多数量有意义的更新。This effect can be seen if one looks carefully at Figure 12.8, which compares the two algorithms on the 19-state random walk task.

![247](https://pic.imgdb.cn/item/60fbd74f5132923bf8fe09e1.png)

## 真实的在线TD($\lambda$)

The online $\lambda$-return algorithm just presented is currently the best performing temporal difference algorithm. It is an ideal which online TD(") only approximates. As presented, however, the online $\lambda$-return algorithm is very complex. Is there a way to invert this forward-view algorithm to produce an efficient backward-view algorithm using eligibility traces？事实上，对于线性函数逼近，的确存在一种在线$\lambda$-回报算法的精确计算实现。之所以叫这个名字是因为它比TD($\lambda$)对在线$\lambda$-回报算法的近似更“真实”...

**真实的在线TD($\lambda$)算法的推导有些复杂（可以参考下一节以及论文 van Seijen et al., 2016的附录）。**

对于$\hat{v}(S_t, \mathbf{w})=\mathbf{w}^T\mathbf{x}(s)$，我们可以得到真实的在线TD($\lambda$)算法

![248](https://pic.imgdb.cn/item/60fbd74f5132923bf8fe09ee.png)

This algorithm has been proven to produce exactly the same sequence of weight vectors, $\mathbf{w}_t$, as the online $\lambda$-return algorithm (van Seijen et al. 2016). 真实在线TD($\lambda$)算法的内存需求与常规的TD($\lambda$)相同，然而每一个的计算量增加了约50%（在资格迹中多出了一个內积）。总体上，每一步的计算复杂度仍然是$O(d)$，与TD($\lambda$)相同。

![249](https://pic.imgdb.cn/item/60fbd74f5132923bf8fe0a04.png)

真实在线TD($\lambda$)算法的资格迹（式12.11）被称为**荷兰迹**（该命名并无特殊算法含义，只是由于算法发展过程中有两位荷兰人对此做出了重要贡献），用以与TD($\lambda$)算法算法中的迹作区分，没有后面一项的迹叫做**积累迹**。早期的研究中经常使用一种被称为替换迹的第三种迹，这种迹的定义只适用于表格型的情况或者特征向量时二值化的情况（如用瓦片编码产生的特征），

![250](https://pic.imgdb.cn/item/60fbd74f5132923bf8fe0a17.png)

如今替换迹已经过时了，荷兰迹几乎可以完全取代替换迹。

## * 蒙特卡洛学习中的荷兰迹

尽管资格迹与时序差分学习在历史发展上是紧密关联的，但实际上它们互相之间并没有关系。对之前介绍的前向视图线性蒙特卡洛算法，利用荷兰迹可以推导出一个等价的但是计算代价更小的后向视图算法。这是唯一明确表示前向视图和后向视图等价的地方。**它有助于真实的在线TD($\lambda$)算法和在线$\lambda$-回报算法等价性的证明。**

![251](https://pic.imgdb.cn/item/60fbd7c35132923bf8ffef82.png)

为了使例子更加简单，我们假设回报值$G$是在幕结束时得到的单一收益回报值，且没有折扣存在。蒙特卡洛算法是一种离线算法，不过这里我们并不是为了解决“离线”更新问题，而是追求一个实现该算法的计算优势。在每幕结束前权重仍然不会更新，但是在幕中的每一步多做一些运算，在最后少做一些运算。这回更加平均地分配计算——每步复杂度为$O(d)$，且这样也消除了在幕结尾使用特征向量而把它存储下来的需求。相反，我们引入资格迹，它总结了到目前为止看到的所有特征向量，这足以在幕结尾高效地精确重构相同的整体蒙特卡洛更新序列。

![252](https://pic.imgdb.cn/item/60fbd7c35132923bf8ffef90.png)

其中，$\mathbf{a}\_{T-1},\mathbf{z}\_{T-1}$​是在T-1时刻两个辅助记忆向量的值，<font color=red>它们可以在不知道G的条件下被增量式地更新，每一个时间步骤的复杂度为$O(d)$​</font>。$\mathbf{z}_t$​实际上是资格迹中的荷兰迹。它用$\mathbf{z}_0=\mathbf{x}_0$​初始化，并根据下式更新

![253](https://pic.imgdb.cn/item/60fbd7c35132923bf8ffefa0.png)

$\mathbf{a}_t,\mathbf{z}_t$在每个时刻t更新，并且在时刻T观测到G时，它们被用来更新计算$\mathbf{w}_T$。In this way we achieve exactly the same final result as the MC/LMS algorithm that has poor computational properties (12.13), but now with an incremental algorithm whose time and memory complexity per step is $O(d)$. 这是一个令人惊讶且吸引人的结果，因为资格迹（特别是荷兰迹）的概念是在没有时序差分学习的情境下被提出的。资格迹看起来并不只是针对时序差分学习的，它们更加基础。**任何时候只要我们想要高效地学习长期的预测，资格迹就有用武之地。**

## Sarsa($\lambda$)

我们只需很少的改动就可以将资格迹扩展到动作价值方法中。

![254](https://pic.imgdb.cn/item/60fbd7c35132923bf8ffefc4.png)

![255](https://pic.imgdb.cn/item/60fbd7c35132923bf8ffefde.png)

第一个图展示了智能体在单幕中选择的道路。在这个例子中，所有的估计价值的初值都设为0，且除了一个用G表示的目标位置的收益为正值外，其他所有的收益都是0. 其他三个图中的箭头分别展示了对于不同的算法，智能体到达目标后哪些动作的价值会增加，增加多少。单步法只会更新最后一个动作价值，n-步伐会均等地更新最后的n个动作价值，资格迹法会从幕的开始不同程度地更新所有的动作价值，更新程度根据时间远近衰减。采用衰减策略一般是最好的做法。

![256](https://pic.imgdb.cn/item/60fbd7e95132923bf8008624.png)

![257](https://pic.imgdb.cn/item/60fbd7e95132923bf8008635.png)

对于在线$\lambda$-回报算法，也有一种动作价值函数对应的版本，被称为真实在线Sarsa($\lambda$)。除了采用动作价值函数特征向量$\mathbf{x}_t=\mathbf{x}(S_t,A_t)$代替状态价值函数特征向量$\mathbf{x}_t=\mathbf{x}(S_t)$外没有任何改动。

![258](https://pic.imgdb.cn/item/60fbd7e95132923bf8008649.png)

## 变量$\lambda$和$\gamma$

为了用更一般化的形式表述最终的算法，有必要将自举法和折扣的常数系数推广到依赖于状态和动作的函数。也就是说，每个时刻会有一个不同的$\lambda$​和$\gamma$​，用$\lambda_t$​和$\gamma_t$​来表示。$\lambda_t \doteq \lambda(S_t,A_t)$​和$\gamma_t \doteq \gamma(S_t)$​。

使用终止函数$\gamma$的影响尤其重大，因为它改变了回报值，即改变了我们想要估计的最基本的随机变量的期望值。现在回报值被更一般地定义为

![259](https://pic.imgdb.cn/item/60fbd7e95132923bf8008660.png)

为了保证和是有限的，我们要求$\Pi_{k=t}^{\infty}\gamma_k=0$对所有t成立。One convenient aspect of this definition is that it enables the episodic setting and its algorithms to be presented in terms of a single stream of experience, without special terminal states, start distributions, or termination times. An erstwhile terminal state becomes a state at which $\gamma(s) = 0$ and which transitions to the start distribution. In that way (and by choosing $\gamma(·)$ as a constant in all other states) we can recover the classical episodic setting as a special case. State dependent termination includes other prediction cases such as pseudo termination, in which we seek to predict a quantity without altering the flow of the Markov process. Discounted returns can be thought of as such a quantity, in which case state-dependent termination unifies the episodic and discounted-continuing cases. (The undiscounted-continuing case still needs some special treatment.)

$\lambda$的变化不像折扣一样是问题本身的变化，而是解决方案的变化。这个泛化影响了状态和动作的$\lambda$-回报值。新的基于状态的$\lambda$-回报值可以被递归得写成

![260](https://pic.imgdb.cn/item/60fbd7e95132923bf800867a.png)

**<font color=red>这个公式要好好理解一下</font>.** 具体是怎么推导出来的我还没明白，书中是指讲了每一个部分的含义。

这个公式表明$\lambda$-回报值是第一个收益值（没有折扣且不受自举操作影响）加上一个可能的第二项（$1-\gamma$可以理解成终止的概率）。当不在下一个状态终止时，我们就会有第二项。根据状态自举的程度，可以将它分成两种情况。在有自举的情况下， 这一项就是状态的估计价值；在无自举的情况下，这一项就是下一时刻的$\lambda$-回报值。

![261](https://pic.imgdb.cn/item/60fbd80c5132923bf8010c71.png)

## 带有控制变量的离轨策略资格迹

最后一步是引入重要度采样。对于完整的非截断$\lambda$-回报，我们没有一个可以引入重要度采样实际可行的方案。因此，我们直接考虑带有控制变量的每次决策型重要度采样的自举法的推广，先回顾一下

![109](https://pic.imgdb.cn/item/60fbc0005132923bf89d6641.png)

类比上式可以得到状态的$\lambda$-回报定义式

![262](https://pic.imgdb.cn/item/60fbd80c5132923bf8010c7a.png)

![263](https://pic.imgdb.cn/item/60fbd80c5132923bf8010c93.png)

![264](https://pic.imgdb.cn/item/60fbd80c5132923bf8010ca8.png)

上述形式的$\lambda$-回报可以很方便地使用前向视图更新，

![265](https://pic.imgdb.cn/item/60fbd80c5132923bf8010cbf.png)

经过变化，可以写成基于资格迹的更新。但这只是前向视图的单个时刻的情况。我们寻找的关系是，沿着各个时刻求和的前向视图更新与沿时刻累加的反向视图更新是近似等价的（因为我们再次忽略了价值函数的变化）。对t求和

![266](https://pic.imgdb.cn/item/60fbd88e5132923bf802f55f.png)

如果可以将第二个求和项中的整个表达式写成资格迹的形式并进行增量的更新，那么上式可以写成后向视图时序差分更新的总和的形式。

![267](https://pic.imgdb.cn/item/60fbd88e5132923bf802f572.png)

我们可以遵循一系列非常相似的步骤获得动作价值函数方法的离轨策略资格迹和对应的通用Sarsa($\lambda$)算法。

![268](https://pic.imgdb.cn/item/60fbd88e5132923bf802f58c.png)

![269](https://pic.imgdb.cn/item/60fbd88e5132923bf802f59f.png)

使用和基于状态的情况完全类似的步骤，可以基于式12.27写出前向视图更新，用求和规则转换更新的求和项，最后得出如下形式的动作价值函数的资格迹

![270](https://pic.imgdb.cn/item/60fbd88e5132923bf802f5c1.png)

![271](https://pic.imgdb.cn/item/60fbd8c85132923bf803cc30.png)

当$\lambda=1$时，这些算法与对应的蒙特卡洛算法密切相关。

![272](https://pic.imgdb.cn/item/60fbd8c85132923bf803cc44.png)

如果$\lambda<1$​，那么所有这些离轨策略算法都涉及自举法和致命三要素，这意味着它们只能在表格型、状态聚合等有限形式的函数逼近下保证稳定。对于线性和更一般的函数逼近形式，参数向量可能会发散到无穷大。

## 从Watkins的$Q(\lambda)$​到树回溯$TB(\lambda)$​

## 采用资格迹保障离轨策略方法的稳定性

**<font color=red>上面两节看得不是很懂，暂时放一放</font>。**

## 实现中的问题

初看上去使用资格迹的表格型方法可能比单步法复杂很多。最朴素的实现需要对每个状态（或状态-动作）在每个时间步骤同时更新价值估计和资格迹。幸运的是，对于$\lambda,\gamma$的一些典型值，几乎所有状态的资格迹都总是接近于0，只有那些最近被访问过的状态的资格迹才会有显著大于0的值，因此，实际上只需要更新这几个状态，就能对原始算法有很好的近似。

有了这个技巧，在表格型方法中使用资格迹的计算复杂度通常只是单步法的几倍。当然确切的背时取决于$\lambda,\gamma$​和其他部分的计算复杂度。请注意，**表格型情况在某种意义上是资格迹计算复杂度的最坏情况**。当使用函数逼近时，不使用资格迹的计算优势通常会减小。

## 小结

This chapter has offered an introduction to the elegant, emerging theoretical understanding of eligibility traces for on and
off-policy learning and for variable bootstrapping and discounting. One aspect of this elegant theory is true online methods, which exactly reproduce the behavior of expensive ideal methods while retaining the computational congeniality of conventional TD methods. Another aspect is the possibility of derivations that automatically convert from intuitive forward-view methods to more efficient incremental backward-view algorithms.

蒙特卡洛法由于不使用自举法，可能在非马尔科夫任务中有优势。因为资格迹使时序差分更向蒙特卡洛法，它们在这些情况下也可以有优势。如果由于其他优点而想要使用时序差分法，但任务至少部分为非马尔科夫，那么就可以使用资格迹方法。资格迹是针对长延迟的收益和非马尔科夫任务的首选方法。

使用资格迹方法需要比单步法进行更多的计算，但作为好处，它们的学习速度显著加快，特别是当收益被延迟许多时间步骤时。因此，在数据缺失并且不能被重复处理的情况下，使用资格迹通常是有意义的，许多在线应用就是这个情况。另一方面，在离线应用中可以很容易的生成数据，例如从模拟中产生，那么通常不使用资格迹。在这些情况下，目标不是从有限的数据中获得更多的信息，而是尽可能地处理尽可能多的数据。使用资格迹带来的学习加速通常抵不上它们的计算成本，因此在这种情况下，单步法更受欢迎。

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
