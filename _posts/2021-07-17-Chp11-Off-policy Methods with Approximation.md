---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter11
subtitle:   Off-policy Methods with Approximation
date:       2021-07-17
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

相较于同轨策略学习，对离轨策略学习进行函数逼近的拓展是截然不同的，而且困难得多。挑战主要来自两方面：一是需要处理更新的目标（这个可以通过importance sampling来应对），第二个挑战是需要处理更新的分布（状态或者状态-动作对的分布），因为离轨策略情况下更新后的分布于同轨策略的分布并不一致。<font color=red> 同轨策略的分布对于半梯度方法的稳定性而言至关重要</font>。有两种通用的方式可以解决这个问题。：一种是再次使用重要度采样，扭曲更新后的分布将其变回同轨策略分布；另一种是**寻找不依赖于任何特殊分布就能稳定的真实梯度算法**。

## 半梯度方法

首先介绍一下半梯度方法在离轨策略中的应用。通过重要度采样解决的第一个挑战，但是没有解决第二个。相应地，这些方法可能在某些情况下会发散。这些方法只对表格型的情况保证稳定且渐近无偏（虽然表格型也可以看成是一种特殊的函数逼近，但是本质上它没有考虑特征之间的相似性）。

之前我们描述了一系列表格型情况下的离轨策略算法。为了把它们转换成半梯度的形式。我们简单地使用近似价值函数（$\hat{v}$或者$\hat{q}$）及其梯度，把对于一个数组（$V$或者$Q$）的跟新替换成对于一个权重向量$\mathbf{w}$的更新。

![189](https://pic.imgdb.cn/item/60fbca3d5132923bf8c8df78.png)

例如，单步状态价值函数的预测算法就是半梯度的离轨策略TD(0)，差别就是添加了一个$\rho_t$

![190](https://pic.imgdb.cn/item/60fbca3d5132923bf8c8df96.png)

$\delta_t$的确切定义取决于这个问题是分幕式、带折扣的，还是持续性、无折扣的。

![191](https://pic.imgdb.cn/item/60fbcd935132923bf8d6deb4.png)

对于动作价值函数，可以用半梯度期望Sarsa来进行拓展

![192](https://pic.imgdb.cn/item/60fbcd935132923bf8d6dec9.png)

注意，这个算法并未使用重要度采样。在表格型的情形下很明显这样做是合适的，因为$S_{t+1}$的随机性被求期望求掉了。但是在使用函数逼近的情况下，就没有这么确定了，因为这是对某一个状态价值的改变可能会影响到其他状态的价值（With function approximation it is less clear because we might want to weight different state–action pairs differently once they all contribute to the same overall approximation.）。

在这些算法的n步推广中，状态价值函数和动作价值函数算法都包括了重要度采样。例如，n步Sarsa

![193](https://pic.imgdb.cn/item/60fbcd935132923bf8d6dedf.png)

<font color=red>但我觉得这里的重要度采样应该要乘到$\rho_{t+n}$，因为$S_{t+n},A_{t+n}$这一步也是要加权的，如果是n步期望Sarsa就对了，但是相应的$G_{t:t+n}$也要改一下</font>。

之前还介绍了一个完全不包含重要度采样的离轨策略算法：n步树回溯算法，以及定义了一个n步$Q(\sigma)$算法，同意了所有的动作价值函数算法，它们相应的半梯度形式也可以类比得到。感兴趣的可以去试一试。

## 离轨策略发散的例子

为了建立直觉，我们考虑一个非常简单的情况。

![194](https://pic.imgdb.cn/item/60fbcd935132923bf8d6defb.png)

假设初始$w=10$，这将会引发从估计价值为10的状态转移到估计状态为20的状态，这看起来是一个好的转移。之后$w$将被增大，以提升第一个状态的估计价值。然后继续更新，$w$将会发散到无穷。

![195](https://pic.imgdb.cn/item/60fbcd935132923bf8d6df0a.png)

如果$1+\alpha(2\gamma-1)$​大于1，那么系统是不稳定的。

这个例子的关键在于：<font color=red>一个转移重复发生，但是$\mathbf{w}$没有在其他转移上更新</font>。这在离轨策略训练的情况下是可能的，因为行动策略可能选择那些目标策略不会选的其他转移所对应的相对动作。对于这些转移而言，$\rho_t$将会是0，而且不会做更新。In the on-policy case the promise of future reward must be kept and the system is kept in check. But in the off-policy case, a promise can be made and then, after taking an action that the target policy never would, forgotten and forgiven.

反正我是觉得这个例子不是那么令人信服，毕竟如果目标策略就只是从1转移到2，那同轨策略你也稳定不了。不过接下来就介绍一个完整的反例——Baird反例。7个状态和2个动作的分幕式MDP。其中点划线动作等概率让系统进入上方6个状态之一，同时实线动作让系统进入第7个状态。

![196](https://pic.imgdb.cn/item/60fbcdc85132923bf8d7c3e1.png)

由于所有转移的收益都为0，所以对于所有的s，其真实的价值函数都是0. 如果$\mathbf{w}=0$，就可以准确地近似。事实上，我们可以得到许多个解，因为权重分量的个数（8个）大于非终止状态的个数（7个）。除此之外，特征向量的集合$\{\mathbf{x}(s): s \in \mathcal{S}\}$是一个线性独立的集合。从所有这些方面来看，这个任务看起来都很适合与线性函数近似的情况。

![197](https://pic.imgdb.cn/item/60fbcdc85132923bf8d7c3fe.png)

但是当我们把半梯度TD(0)应用到这个问题上时，那么权重会发散到无穷。如上图左边所示，无论步长多么小，不稳定性对于任何正的步长都会发生。事实上，它甚至会在动态规划所采用的期望更新中出现，如上图右边所示。这也就是说，如果权重向量用一种半梯度的方式，对同一时刻的所有状态更新，使用DP（基于期望的）目标

![198](https://pic.imgdb.cn/item/60fbcdc85132923bf8d7c424.png)

在这种情况下，没有随机性也没有异步性，只是一个经典的DP更新。除了使用半梯度的函数近似之外，这个方法非常传统。然而系统还是不稳定。

> 点评一下，这个例子里可以比较好地说明之前那个简单例子中所谓的“一个转移重复发生，但是$\mathbf{w}$没有在其他转移上更新”。由于初始化的原因，导致最下面这个状态的价值是最大的，那么在开始的时候，从其他上面的状态转移到下面这个状态时，就会增加上面相应状态的权重系数，而又由于每个状态都有$w_8$，并且下面这个前面还有个2，所以每次增加上面上面的某个状态的权重时，下面这个状态的价值也会增加，还增加的更多。而且由于不会出现从下面到上面以及上面状态之间的转移更新（$\rho_t=0$），所以$w_8$就会一直增大。

如果我们只是在Baird反例中更换DP的更新分布：将均匀分布更换为同轨策略分布（通常需要异步更新），那么就可以保证收敛（<font color=red>这我是有疑惑的，同轨策略不就是目标策略吗，那不是依然还是有上面说的这个问题？</font>）。这个例子表明，如果没有按照同轨策略分布更新，即使是最简单的自举法和函数逼近法的组合也有可能是不稳定的。

在Q-learning中，有许多类似于Baird所示的发散的反例。Considerable effort has gone into trying to find a remedy to this problem or to obtain some weaker, but still workable, guarantee. 例如，只要行动策略和目标策略足够接近，可能可以保证Q_learning的收敛性，比如在$\epsilon$-贪心策略下。To the best of our knowledge, Q-learning has never been found to diverge in this case, but there has been no theoretical analysis.

假设我们不是像在Baird的反例中那样在每一步迭代里想期望的单步回报走一步，而是事实上一路都在向最好的最小二乘近似去优化价值函数，者能够解决不稳定问题吗？

如果特征向量$\{\mathbf{x}(s): s \in \mathcal{S}\}$组成一组线性无关的集合，那么是可以的，因为此时在每个迭代中都可以得到精确解，这也可以规约到标准的表格型DP中。但此处的关键问题是考虑精确解不存在的情况（比如线性函数逼近一个非线性的函数）。在这种情况下，即使在每次迭代都得到了最好的近似，也是不能保证稳定性的，下面这个例子展示了线性函数逼近可能不适用于DP，即使在每一步都找到了最小二乘解。

![199](https://pic.imgdb.cn/item/60fbcdc85132923bf8d7c436.png)

> 这里出问题的原因就在于状态表征向量是线性相关的。

另一个避免不稳定的方式是使用函数逼近中的一些特殊的方法。In particular, stability is guaranteed for function approximation methods that do not extrapolate from the observed targets. These methods, called *averagers*, include nearest neighbor methods and locally weighted regression。

## The Deadly Triad

The danger of instability and divergence arises whenever we combine all of the following three elements, making up
what we call the deadly triad :

- **Function approximation** A powerful, scalable way of generalizing from a state space much larger than the memory and computational resources (e.g., linear function approximation or ANNs).
- **Bootstrapping** Update targets that include existing estimates (as in dynamic programming or TD methods) rather than relying exclusely on actual rewards and complete returns (as in MC methods).
- **Off-policy training** Training on a distribution of transitions other than that produced by the target policy. 如动态规划中所做的，遍历整个状态空间并且均匀地更新所有状态而不理会目标策略就是一个离轨策略训练的例子。

特别要注意，这种风险并不是由于控制或者广义策略迭代造成的。在比控制更简单的预测问题中，只要包含致命三要素，不稳定性也会出现。

如果致命三要素中只有两个要素是满足的，那么可以避免不稳定性。很自然的，我们会考虑哪一个要素是可以去掉的。

首先，函数近似是最不可能被舍弃的。我们需要能处理大规模问题并具有足够表达能力的方法。

不使用自举法时有可能的，付出的代价是计算和数据上的效率。自举法通常可以提高学习的速度，因为它允许学习过程中利用状态的性质和自举计算中识别状态的能力。另一方面，自举法会削弱某些问题上的学习：在这些问题中状态的表征不好，所以泛化性很差。一个不好的状态表征会产生偏差，这也是导致自举法的近似边界更差的原因。

最后一个是离轨策略学习。同轨策略的方法通常来说是足够了。对于无模型的强化学习而言，可以简单地使用Sarsa而不是Q-learning。离轨策略方法将行动策略与目标策略分离，这可以被视作一个吸引人的便利，但并不是必要的。不过它对于构建强人工智能这个更大的目标而言可能很重要。这时候智能体往往不止学习一个价值函数和一个策略，而是同时学习多个。目标策略有多个，因此一个行动策略不可能同时等同于所有目标策略。

## 线性价值函数的几何性质

我们想象一个由所有可能的状态价值函数构成的空间，所有的函数都把状态映射到一个实数：$v:\mathcal{S} \to \mathbb{R}$。这些价值函数中很多都不对应于任何一个策略。更重要的是，很多都不可以被函数逼近器表示。

给定一个状态空间的枚举向量$\mathcal{S}=\{s_1,s_2,\ldots,s_{\|\mathcal{S}\|}\}$​​，任何价值函数$v$​​都对应于一个价值向量$[v(s_1), v_{s_2},\ldots,v_{s_{\|\mathcal{S}\|}}]^T$​​。在多数情况下，我们想使用函数逼近方法，因为显式表达出这个向量需要的元素数量太多。然而，这个向量的想法在概念上是十分有用的，我们认为一个价值函数和它的向量表示式可互换的。

为了可视化，我们考虑这样一种情况：有三个状态$\mathcal{S}=\{s_1,s_2,s_3\}$和两个参数$\mathbf{w}=(w_1,w_2)^T$，我们可以把所有价值函数/价值向量当做三维空间中的点。这些参数构成了在二维子空间上的另一个坐标系统。任何权重向量是一个二维子空间上的点，这些点构成了近似价值函数$v_{\mathbf{w}}$的空间。

现在我们考虑一个固定的策略$\pi$，我们假设它的真实价值函数太复杂，不能用一个近似函数来精确表示。因此$v_{\pi}$不在这个子空间中。在下图中被描绘为可表示函数组成的子空间（平面）的上方。

![200](https://pic.imgdb.cn/item/60fbcdc85132923bf8d7c450.png)

既然$v_{\pi}$不能被完全表示，那么距离它最近的价值函数是什么呢？一般我们用欧式距离去度量两个向量之间的距离，不过在强化学习中，还要考虑状态的重要程度，有些状态会出现得更频繁。所以和之前一样，我们使用分布$\mu:\mathcal{S} \to [0,1]$来量化我们对不同状态的关注程度（通常使用同轨策略分布），然后用如下范数来定义价值函数之间的距离

![201](https://pic.imgdb.cn/item/60fbcdff5132923bf8d8b90d.png)

之前介绍的$\overline{VE}(\mathbf{w})=\Vert v_{\mathbf{w}} - v_{\pi}\Vert_{\mu}^2$。对于任何价值函数$v$，用来在可表示的函数子空间内寻找其最近邻的函数的操作就是投影操作。定义一个投影算子$\Pi$：

![202](https://pic.imgdb.cn/item/60fbcdff5132923bf8d8b920.png)

$\Pi_{v_{\pi}}$是可以被蒙特卡洛方法找到的近似解，尽管通常很慢。

![203](https://pic.imgdb.cn/item/60fbcdff5132923bf8d8b934.png)

TD方法会找到不同的解。为了理解它们的基本原理，回顾一下贝尔曼方程

![204](https://pic.imgdb.cn/item/60fbcdff5132923bf8d8b948.png)

真实价值函数$v_{\pi}$是唯一满足上式的价值函数。如果用一个近似价值函数$v_{\mathbf{w}}$去替换$v_{\pi}$，则改变过的方程左右两侧的差值可以用于衡量函数$v_{\mathbf{w}}$和$v_{\pi}$之间的差距。定义为在状态$s$的**贝尔曼误差**

![205](https://pic.imgdb.cn/item/60fbcdff5132923bf8d8b967.png)

可以看到，贝尔曼误差是TD误差的期望。

在所有状态下的贝尔曼误差组成的向量$\bar{\delta}_{\mathbf{w}} \in \mathbb{R}^{\|\mathcal{S}\|}$​，被称作贝尔曼误差向量（在图11.3中用BE表示）。这个向量在范数上的总大小，是价值函数的误差的一个总衡量，称作贝尔曼均方误差

![206](https://pic.imgdb.cn/item/60fbcf3c5132923bf8ddd8ab.png)

通常可能把$\overline{BE}$减到0，但是对于线性函数逼近，有一个唯一的最小化$\overline{BE}$的值$\mathbf{w}$。这个点在函数子空间中（图中用$\min \overline{BE}$标记）通常不同于最小化$\overline{VE}$的点（用$\Pi_{v_{\pi}}$表示）。

贝尔曼误差向量就是将贝尔曼算子作用在近似价值函数中的结果。

![207](https://pic.imgdb.cn/item/60fbcf3c5132923bf8ddd8c1.png)

如果把贝尔曼算子作用到一个可表示空间的价值函数上，那么通常它会产生一个在子空间之外的价值函数，如图中所示。在动态规划（不包含函数逼近）中，这个算子会反复作用到可表示空间之外的点，如图中灰色箭头所指示的。最终收敛到真实的价值函数$v_{\pi}$：贝尔曼算子的唯一不动点。

然而，在有函数逼近的情况下，无法表示空间之外的中间过渡价值函数。因为在第一次更新后，价值函数必须被投影回可表示空间，之后再在可表示空间进行下一次贝尔曼迭代；价值函数又一次被带出子空间，然后又被投影回子空间。

在这种情况下，我们感兴趣的是贝尔曼误差向量在可表示空间内的投影。

![208](https://pic.imgdb.cn/item/60fbcf3c5132923bf8ddd8d0.png)

使用线性函数逼近，总会存在一个近似的价值函数（在子空间中）使得$\overline{PBE}$为0。这是一个TD的不动点。这个点在半梯度的TD方法和离轨策略训练（同时）下并不总是稳定的。这个价值函数通常也不同于那些优化$\overline{VE}$所和$\overline{BE}$得到的的价值函数。

## 对贝尔曼误差做梯度下降

到目前为止，只有蒙特卡洛方法是真正的SGD。这些方法在同轨策略和离轨策略以及一般的非线性（可导）函数逼近器下均可有效地收敛，尽管它们通常慢于使用自举法的半梯度方法（非SGD）。而半梯度方法可能在离轨策略学习中发散。

The appeal of SGD is so strong that great effort has gone into finding a practical way of harnessing it for reinforcement learning. 我们首先应该考虑的是待优化的误差或者目标函数的选择。接下来讨论将贝尔曼误差作为目标函数。Although this has been a
popular and influential approach, the conclusion that we reach here is that it is a misstep and yields no good learning algorithms.

首先我们考虑更简单一些的情况，贝尔曼误差是TD误差的期望，那么我们先考虑TD误差。把最小化TD误差平方的期望作为目标——均方TD误差

![209](https://pic.imgdb.cn/item/60fbcf3c5132923bf8ddd8e3.png)

因此，按照SGD的方法，我们可以得到基于这种期望值的一个样本的单步更新。

![210](https://pic.imgdb.cn/item/60fbcf3c5132923bf8ddd8f9.png)

可以发现，除了额外多出了最后一项，它与半梯度TD算法无异。这一项补全了这个梯度，而让他成为了一个真正的SGD算法并有优异的收敛保证。这个算法叫做朴素残差梯度法（naive residual-gradient algorithm (after Baird, 1995)）。

**尽管朴素残差梯度法稳健地收敛，但是它并不一定会收敛到我们想要的地方。**

![211](https://pic.imgdb.cn/item/60fbcf855132923bf8df0a5a.png)

在A-分裂的例子中使用了表格型的表示，因此真实的价值函数可以被确切地表达出来，然而朴素残差梯度法发现了不同的值，而这些值相比于真实值有更低的$\overline{TDE}$。**<font color=red>最小化$\overline{TDE}$是朴素的，通过减小所有的TD误差，它更倾向于得到时序上平滑的结果，而不是准确的预测</font>。**

接下来就考虑贝尔曼误差。如果学到了准确的价值，那么在任何位置上贝尔曼误差都是0。因此，最小化贝尔曼误差在A-分裂的例子中应该没有任何问题。通常我们不能期望达到0的贝尔曼误差，因为我们一般认为真实的价值函数在可表达的函数空间之外。

![212](https://pic.imgdb.cn/item/60fbcf855132923bf8df0a69.png)

所有的期望都隐式地以$S_t$为条件，这种更新和采样它的多种方式被称为残差梯度算法。如果简单地在所有期望中使用采样值，那么上面的公式就几乎规约到朴素残差梯度法。（对于状态价值而言，在处理$\rho_t$时有点不同，在类似动作价值的情况里，可以完全规约）

但是这种做法太naive了，因为公式里包括了下一个状态$S_{t+1}$，其在两个相乘的期望中出现。为了得到这个乘积的无偏样本，需要下一个状态的两个独立样本，但是通常在与外部环境的交互过程中，我们只能得到一个。可以一个用期望值，一个用采样值。

有两种方法可以让残差梯度法起作用。一种是在确定性的环境下，如果到下一个状态的转移是确定性的，那么两个样本一定是一样的，则朴素的算法就有效。另一种方法就是从$S_t$中获得下一个状态$S_{t+1}$的两个独立的样本。In real interaction with an environment, this would not seem possible, but when interacting with a simulated environment, it is.  In either of these cases the residual-gradient algorithm is guaranteed to converge to a minimum of the $\overline{BE}$ under the usual conditions on the step-size parameter.

然而，残差梯度法的收敛性仍然有三点不能让人满意。低一点是在实际应用中它速度非常慢。第二点是它看起来仍然收敛到错误的值。它在所有表格型情况下都能找到正确的值，因为在那些情况下贝尔曼方程都能找到精确的解。但是如果我们使用真正的函数逼近来检查样本时，我们会发现残差梯度法似乎找到了错误的价值函数。One of the most telling such examples is the variation on the A-split example known as the A-presplit example, shown on the preceding page, in which the residual-gradient algorithm finds the same poor solution as its naive version.

![213](https://pic.imgdb.cn/item/60fbcf855132923bf8df0a87.png)

这里关键点是：**算法只关注特征**。所以它并不知道实际上有$A_1$​和$A_2$​两个状态，而只是通过特征认为只有一个状态$A$​

残差梯度法第三个不令人满意的点是贝尔曼误差是不可学习的。

## 贝尔曼误差是不可学习的

这里的不可学习和机器学习中的概念是不同的。机器学习中，可学习指的是它可以被高效地学习，意味着可以用多项式级而不是指数级数量的样本学会它。在这里不可学习的概念则更为基础，不可学习意味着用任意多的经验也学习不到。

事实上，很多在强化学习中我们明显感兴趣的量，即使有无限多的数据，也不能从中学到。这些量都是良好定义的，而且在给定环境的内在结构时可以计算出来，但是不能从外部可观测的特征向量中得到。（如果能够观测到状态序列而不仅仅是相应的特征向量，那么当然可以估计它们）

考虑两个马尔科夫收益过程，如下图

![214](https://pic.imgdb.cn/item/60fbcf855132923bf8df0a9e.png)

当两条边离开同一个状态时，两个转移都被认为是等概率发生的，数字表明了获得的收益。所有状态看起来都是一样的，它们有相同的单变量特征$x=1$而且有相同的近似值$w$。因此，数据轨迹唯一不同的部分是收益序列。左侧MRP处于同一个状态下，随机等概率产生无穷的0和2的流。右侧MRP，在每一步中，从左侧的状态出发收益是0，右侧出发收益是2，由于在每一步中处于这两个状态的概率都是一样的，可观测的数据又一次是由一个0和2随机组成的无限长的流。因此即使给定了无限量的数据，仍然不可能分辨是哪一个MRP产生了它。In particular, we could not tell if the MRP has one state or two, is stochastic or deterministic. These things are not learnable.

这一对MRP也展示了$\overline{VE}$是不可学习的。如果$\gamma=0$，then the true values of the three states (in both MRPs), left to right, are 1, 0, and 2. 假设$w=1$，那么$\overline{VE}$对左边而言是0，但对右侧而言是1. $\overline{VE}$在这两个问题中使不一样的，但是产生的数据却遵从同一个分布，因此$\overline{VE}$也是不可学习的。

如果一个目标不可学习，那么它的效用确实会受到质疑。Note that the same solution, w = 1, is optimal for both MRPs above (assuming μ is the same for the two indistinguishable states in the right MRP). Is this a coincidence, or could it be generally true that all MDPs with the same data distribution also have the same optimal parameter vector?

**<font color=red>$\overline{VE}$是不可学习的，但是优化的它的参数是可学习的。</font>**

为了理解这一点，我们引入一个明显是可以学习的量。一个总是可观察的误差是每个时刻的估计价值与这个时刻之后的实际回报的误差。*均方回报误差*，又称为$\overline{RE}$.

![215](https://pic.imgdb.cn/item/60fbcf865132923bf8df0ace.png)

最后一项是不依赖于$\mathbf{w}$的，所以这两个目标具有相同的最优参数。

现在回到$\overline{BE}$，和$\overline{VE}$类似，可以从MDP的知识中计算它但是不可以从数据中学出来。但是它与$\overline{VE}$的不同在于，它的极小值也是不可学习的。下面是一个反例。

![216](https://pic.imgdb.cn/item/60fbd02a5132923bf8e1ba05.png)

这个例子里关于第二个MDP最小化$\overline{BE}$的解$\mathbf{w}=(-\frac{1}{2},0)^T$，我是有一些疑惑的。我自己算了一下，过程我放在下面，如果有问题欢迎大家指正。

<img src="https://pic.imgdb.cn/item/60fbd02a5132923bf8e1ba2c.png" alt="218" style="zoom:50%;" />

因此，$\overline{BE}$是不可学习的，不能从特征向量和其他可观测的数据中估计它。这限制了$\overline{BE}$用于有模型的情形。几乎没有算法能够在不接触特征向量以外的底层MDP状态的情况下最小化$\overline{BE}$。

残差梯度法是唯一能够最小化$\overline{BE}$的算法，因为它允许从相同的状态中采样两次——这些状态不仅仅是特征向量相同，而且是保证有完全相同的基础底层状态。

下图总结了一般的情况。

![217](https://pic.imgdb.cn/item/60fbd02a5132923bf8e1ba16.png)

> 我觉得这里的不可学习有点牵强了，不过如果针对是的强化学习问题，那么还是有些道理的。因为在强化学习中状态是一个极其重要的量，所以如果状态不能被准确表达的话，对强化学习是用很大的影响的。一句话概括：在真实价值函数不在近似函数空间中时，所有和状态（不是特征）真实价值相关的目标函数都是不可学习的。

## 梯度TD方法

我们现在考虑最小化投影贝尔曼误差（$\overline{PBE}$）的SGD方法。在线性情况下，总存在一个精确的解——TD的不动点$\mathbf{w}_{TD}$，这一点上$\overline{PBE}=0$. 这个解可以通过最小二乘法找到，但是只能用时间复杂度是参数数量平方级$O(d^2)$的方法。我们现在寻找的时间复杂度为$O(d)$并且具有稳定收敛性质的SGD方法。

我们以矩阵形式重写目标函数

![219](https://pic.imgdb.cn/item/60fbd02a5132923bf8e1ba48.png)

为了把它转化成SGD方法，我们需要在每个时间点上采样一些东西，并把这个量作为它的期望值的估计。比如最后一个因子

![220](https://pic.imgdb.cn/item/60fbd02a5132923bf8e1ba59.png)

这就是半梯度TD(0)更新式的期望。第一个因子就是这个更新梯度的转置

![221](https://pic.imgdb.cn/item/60fbd0e95132923bf8e4b240.png)

最后，中间的因子是特征向量的外积矩阵的期望的逆

![222](https://pic.imgdb.cn/item/60fbd0e95132923bf8e4b256.png)

我们把$\overline{PBE}$梯度的三个因子换成这些期望的形式，得到

![223](https://pic.imgdb.cn/item/60fbd0ea5132923bf8e4b26a.png)

把梯度写成这样可能看不出有什么好处。这是三个期望的乘积（期望默认是给定$\mathbf{x}\_t$​​下的条件期望），第一个和第三个不是独立的，它们都依赖于下一个特征向量$\mathbf{x}\_{t+1}$​​，我们不能仅仅采样一次样，然后将这些样本相乘。这会像朴素残差梯度算法那样得到一个有偏的估计。

另一个想法是分别估计这三个期望，然后再乘起来。这种方法虽然有效，但是需要太对计算资源，尤其是存储前两个期望，需要$d \times d$的矩阵，并且需要计算第二个矩阵的逆。可以改进这个想法。If two of the three expectations are estimated and stored, then the third could be sampled and used in conjunction with the two stored quantities.

分别存储一些估计，然后把它们与采样计算出来的值合并，这是一个好的想法，梯度TD方法就是估计并存储后面两个因子，$d \times d$的矩阵和$d$维的向量，因此它们的乘积只是一个$d$维的向量。我们把这个学到的向量记为$\mathbf{v}$

![224](https://pic.imgdb.cn/item/60fbd0ea5132923bf8e4b27d.png)

这个形式就是从特征$\mathbf{x}_t$​近似$\rho_t \delta_t$​的最小二乘的解。可以通过最小化平方误差$(\mathbf{v}^T\mathbf{x}_t - \rho_t \delta_t)^2$​增量式地寻找向量$\mathbf{v}$​的标准SGD方法来获得（**此处增加了一个重要度采样比，为什么加我没搞明白**，<font color=red>我后来在程序里试了一下把$rho_t$放到括号里，最终结果是类似的，但是在训练前期波动就非常大</font>）。

![225](https://pic.imgdb.cn/item/60fbd0ea5132923bf8e4b295.png)

其中，$\beta$是另一个步长，这样我们就可以以$O(d)$空间和时间复杂度有效的得到$\mathbf{v}$。

然后对第一个因子用采样的结果去替代，我们就得到了一个可用的更新规则

![226](https://pic.imgdb.cn/item/60fbd20f5132923bf8e92cc7.png)

这个算法又被称作GTD2. 如果最后的內积$\mathbf{x}_t^T\mathbf{v}_t$是最先完成的，那么整个算法的时间复杂度是$O(d)$。

在替换为$\mathbf{v}_t$前多做几步分析，可以得到一个更好的算法，从式11.29开始

![227](https://pic.imgdb.cn/item/60fbd20f5132923bf8e92cd0.png)

这个算法又被称为带梯度修正的TD(0)（TDC）或者GTD(0)。

下图展示了一个TDC在Baird的反例上的运行的结果和期望的行为。正如所预料的那样，$\overline{PBE}$降到了0，但是参数向量的各个分量并没有接近0. 事实上，这些值仍然距离最优解非常远。**<font color=red>对于所有$s$，$\mathbf{w}$需要与$(1,1,1,1,1,1,4,-2)^T$成正比 ???</font>。**

![228](https://pic.imgdb.cn/item/60fbd20f5132923bf8e92ce0.png)

GTD2和TDC都包含两个学习过程。主要过程学习$\mathbf{w}$，次要过程学习$\mathbf{v}$。主要学习过程依赖的逻辑依赖于次要的学习过程结束，至少是近似结束，然后次要学习过程会继续进行，并且不受主要过程的影响。我们把这种不对称的依赖称为梯级（cascade）。在梯级中我们通常假设次要学习过程要进行得更快，因此总是处于它的渐近值，其足够精确而且已经准备好辅助主要学习过程。这些方法的收敛性证明通常需要显式地做这个假设。这些被称作两个时间量级（two-time-scales）的证明。如果$\alpha$是主要过程的步长，$\beta$是次要过程的步长，那么通常需要限制$\beta \to 0$而且$\alpha/\beta \to 0$。

梯度TD方法目前是最容易理解而且应用广泛的离轨策略方法。

## (强调)Emphatic-TD Methods

现在来看另一种主流的方法。在离轨策略学习中，我们使用重要度采样重新分配了状态转移的权重，使得它们变得适合学习目标策略。但是状态分布仍然是行动策略产生的，这里就有了一个不匹配之处。一个自然的想法是以某种方式重新分配状态的权重，强调一部分而淡化另一部分，目的是将更新分布变为同轨分布。

事实上，“同轨策略分布”的说法并不准确，因为存在有许多同轨策略分布（由于初始状态不同），其中任何一个都能保证稳定。考虑一个无折扣的分分幕问题。无论幕如何开始，如果所有的状态转移都遵从目标策略，那么得到的状态分布一定是同轨策略分布。

如果带折扣，可以定义为在每一步以$1-\gamma$的概率终止并重启的连续操作。这种思考折扣的方式是一种更通用的概念**伪终止**的例子——“终止”不影响状态转移序列，但是影响学习过程和被学习的量。

**<font color=red>下面这段我没有很理解什么意思 </font>。**

这种伪终止对离轨策略学习非常重要，因为重启是可选择的（记住，我们可以以任何喜欢的方式开始）。而且终止减弱了同轨策略分布中对持续记录经过的状态的需求。这也就是说，如果我们不考虑在新状态上重启，那么折扣会很快给我们一个受限的同轨策略分布。

This kind of pseudo termination is important to off-policy learning because the restarting is optional—remember we can start any way we want to—and the termination relieves the need to keep including encountered states within the on-policy distribution. That is, if we don’t consider the new states as restarts, then discounting quickly gives us a limited on-policy distribution.

学习分幕式状态价值的单步强调TD算法定义如下

![229](https://pic.imgdb.cn/item/60fbd20f5132923bf8e92cf8.png)

其中，$I_t$为兴趣值，可以取任意值，而$M_t$为强调值，被初始化为$M_{t-1}=0$。

下图展示了这个算法在Baird反例上的表现结果（对于所有$t$上，$I_t=1$）。这些轨迹是通过迭代地计算参数向量轨迹的期望得到的，而没有任何由于状态转移和收益采样引起的方差。这里没有展示直接使用强调TD算法的结果，因为它在Baird反例上方差过高，所以几乎不可能在计算实验上得到一致性的结果。算法理论上收敛到最优解，但是实际上并不是这样。

![230](https://pic.imgdb.cn/item/60fbd20f5132923bf8e92d0b.png)

## Summary

我们需求离轨策略算法的一个原因是，在处理试探和开发之间的权衡时提供灵活性。另一个是将行为独立于学习，并且避免目标策略的专制。

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
