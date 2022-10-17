---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter9
subtitle:   On-policy Prediction with Approximation
date:       2021-06-24
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

近似函数不再表示成一个表格， 而是一个具有权值向量$\mathbf{w} \in \mathbb{R}^d$​的参数化函数。通常来说，权值的数量（$\mathbf{w}$​的维度）远远小于状态的数量（$d \ll \|S\|$​）。另外改变一个权值将会改变许多状态的估计值。

将强化学习扩展到函数逼近还有助于解决“部分观测”问题，即智能体无法观测到完整的状态的情况。如果参数化函数$\hat{v}$的形式化表达不依赖于状态的某些特定的部分，那么状态的这些部分是不可观测的也不影响整个问题的解决。但是，函数逼近无法根据过往的部分观测记忆来对状态表达进行扩充。

## 价值函数近似

![143](https://pic.imgdb.cn/item/60fbc2b15132923bf8a939f3.png)

Up to now, the actual update has been trivial: the table entry for $s$’s estimated value has simply been shifted a fraction of the way toward $u$,​ and the estimated values of all other states were left unchanged. Now we permit arbitrarily complex and sophisticated methods to implement the update, and updating at $s$ generalizes so that the estimated values of many other states are changed as well.

我们可以将每一次更新都视为一个常规的训练样本，这使得我们可以使用几乎所有现成的函数逼近方法来进行价值函数预测。然而，并不是所有的函数逼近方法都适用于强化学习。这些方法往往都假设存在一个静态训练集，并在此集合上可以进行多次训练。然而，在强化学习中，能够在线学习是很重要的，即智能体需要与环境或者环境的模型进行交互。想要做到这一点，就需要算法能够从逐步得到的数据中有效的学习。<font color=red>此外，强化学习通常需要能够处理非平稳目标函数的函数逼近算法</font>。例如，在基于GPI（广义策略迭代）的控制方法中，我们经常需要在$\pi$变化时学习$q_{\pi}$。即使策略保持不变，使用自举法（DP以及TD学习）产生的训练样例也是非平稳的。

## 预测目标

在表格型情况下，学习的价值函数可以完全与真实的价值函数精确相等。而且，每个状态学习的价值函数是解耦的——一个状态的更新不会影响其他状态。但是在函数逼近中，一个状态的更新会影响许多其他的状态，而且不可能让所有状态的价值函数完全正确。By assumption we have far more states than weights, so making one state’s estimate more accurate invariably means making others’ less accurate. 我们必须指定一个状态的分布$\mu(s) \geq 0, \sum_s \mu(s)=1$来表示我们对于每个状态$s$的误差的重视程度。均方价值误差

![144](https://pic.imgdb.cn/item/60fbc2b15132923bf8a93a0b.png)

通常情况下，我们将在状态$s$​上消耗的计算时间的比例确定义为$\mu(s)$​。在同轨策略训练中称其为同轨策略分布。在持续性任务中，同轨策略分布是$\pi$​下的平稳分布。在分幕式任务中，同轨策略分布于上文有些区别，因为它与幕的初始状态的选取有关。

![145](https://pic.imgdb.cn/item/60fbc2b15132923bf8a93a26.png)

在持续性和分幕式任务的两种情况下智能体的行为是类似的，但是在进行近似求解的时候必须将它们分开来进行严格讨论，这使得学习目标的设定更加规范。

It is not completely clear that the VE is the right performance objective for reinforcement learning. <font color=red>Remember that our ultimate purpose—the reason we are learning a value function—is to find a better policy</font>. The best value function for this purpose is not necessarily the best for minimizing VE. Nevertheless, it is not yet clear what a more useful alternative goal for value prediction might be.

## 随机梯度和半梯度方法

我们假设状态在样本中的分布$\mu$和$\overline{VE}$中的分布相同。在这种情况下，一个好的策略就是尽量减少观测到的样本的误差。

![146](https://pic.imgdb.cn/item/60fbc5ce5132923bf8b67453.png)

SGD在梯度方向只更新一小步的原因并不是很明显。我们能够沿着这个方向移动，知道完全消除这个样本的误差呢？在大多数情况下，这是可以做到的，但这通常是不可取的。<font color=red>我们不寻求或期望找到一个对所有状态都具有零误差的价值函数，我们只是期望找到能够在不同状态平衡其误差的一个近似的价值函数</font>。如果我们在一个时刻中完全修正了每个样本，那么我们就无法找到这样的平衡了。In fact, the convergence results for SGD methods assume that $\alpha$ decreases over time. If it decreases in such a way as to satisfy the standard stochastic approximation conditions, then the SGD method is guaranteed to converge to a local optimum.

现在我们讨论一种新情况：第$t$个训练样本$S_t \to U_t$的目标输出$U_t \in \mathbb{R}$不是真实的价值$v_{\pi}(S_t)$，而是它的一个近似。例如，$U_t$可能是$v_{\pi}(S_t)$的一个带噪版本，或者是用$\hat{v}$获得的一个基于自举法的目标。我们用$U_t$替代$v_{\pi}(S_t)$，就产生了用于进行状态价值函数预测的通用SGD方法

![147](https://pic.imgdb.cn/item/60fbc5ce5132923bf8b67463.png)

如果$U_t$​是一个无偏估计，即满足对于任意的$t$​，有$\mathbb{E}[U_t\|S_t=s]=v_{\pi}(S_t)$​，那么在$\alpha$​满足随机近似条件的情况下，$\mathbf{w}_t$​一定会收敛到一个局部最优解。

如果使用$v_{\pi}(S_t)$的自举估计值作为$U_t$，则无法得到相同的收敛性保证。自举目标，取决于权值向量$\mathbf{w}_t$的当前值，这意味着它们是有偏的。自举法并不是真正的梯度下降的实例，它只考虑了改变权值向量$\mathbf{w}_t$对估计的影响，却忽略了对目标的影响。由于只包含一部分梯度，我们称之为**半梯度方法**。

Although semi-gradient (bootstrapping) methods do not converge as robustly as gradient methods, they do converge reliably in important cases such as the linear case。

Moreover, they offer important advantages that make them often clearly preferred. One reason for this is that they typically enable significantly faster learning. Another is that they enable learning to be continual and online, without waiting for the end of an episode.

状态聚合（State aggregation）是一种简单形式的泛化函数逼近（名字起得这么高端，其实就是piece-wise constant）。这里状态被分成不同的组，每个组对应一个估计值（对应权重向量的一个维度分量）。状态聚合是SGD的一个特殊情况，其梯度对于$S_t$所在的组为1，对其余组为0.

![148](https://pic.imgdb.cn/item/60fbc5ce5132923bf8b67474.png)

下图显示了这个任务真实的价值函数。它几乎是一条直线，只在头尾两端（100个状态）有些弯曲，向水平方向倾斜。运行了100000幕，$\alpha=2 \times 10^{-5}$。对于状态集合，1000个状态分成10组。

![149](https://pic.imgdb.cn/item/60fbc5ce5132923bf8b6748b.png)

通过参考该任务的状态分布$\mu$，会注意到近似价值函数的一些细节。

中心状态500是每幕的第一个状态，但是很少再次被访问到。On average, about 1.37% of the time steps are spent in the start state. The states reachable in one step from the start state are the second most visited, with about 0.17% of the time steps being spent in each of them. From there $\mu$ falls off almost linearly, reaching about 0.0147% at the extreme states 1 and 1000. 该分布最明显的效果体现在：<font color=red>**最左边的组的估计值明显高于真实状态价值的不带权平均值，而最右边的组，其估计值明显偏低**</font>。这是由于这一部分的状态在$\mu$的权值分布上最不对称。例如，在最左边的一组中，状态100的权重比起状态1要大3倍。因此，组的估计偏向于状态100的真实价值，而这个估计高于状态1的真实价值。

## 线性方法

这里的线性指的是权值向量的线性函数。

![150](https://pic.imgdb.cn/item/60fbc5ce5132923bf8b674a6.png)

In particular, in the linear case there is only one optimum (or, in degenerate cases, one set of equally good optima), and thus any method that is guaranteed to converge to or near a local optimum is automatically guaranteed to converge to or near the global optimum. For example, the gradient Monte Carlo algorithm presented in the previous section converges to the global optimum of the $\overline{VE}$ under linear function approximation if $\alpha$ is reduced over time according to the usual conditions.

半梯度TD(0)算法也在线性函数逼近下收敛，但它并不遵从SGD的一般通用结果，对此，我们需要一个额外的定理来证明。权值向量也不是收敛到全局最优，而是靠近局部最优的点。

![151](https://pic.imgdb.cn/item/60fbc6095132923bf8b762d5.png)

![152](https://pic.imgdb.cn/item/60fbc6095132923bf8b762ef.png)

如果需要证明概率版本的收敛性，则需要考虑一些额外情况以及$\alpha$随时间减小的方式。

在TD不动点处，已经证明（在持续性任务的情况下）$\overline{VE}$在可能的最下误差的一个扩展边界内（upper bound）

![153](https://pic.imgdb.cn/item/60fbc6095132923bf8b76303.png)

因为$\gamma$常常接近于1，会这个扩展因子可能相当大，所以TD方法在渐近性能上有很大的潜在损失。但是于MC相比，TD的方差大大减小，因此速度更快。所以哪种方法更好取决于近似和问题的性质，以及学习能够持续多久。

![154](https://pic.imgdb.cn/item/60fbc6095132923bf8b76314.png)



**<font color=red>得到这些收敛结果的关键在于状态是按照同轨策略分布来更新的</font>**。对于按照其他分布的更新，使用函数逼近的自举法可能发散到无穷大。

![155](https://pic.imgdb.cn/item/60fbc6095132923bf8b76338.png)

## 线性方法的特征构造

Linear methods are interesting because of their convergence guarantees, but also because in practice they can be very efficient in terms of both data and computation.  然而是否会具有这些优势很大程度上取决于我们如何选取用来表达状态的特征。选择合适于任务的特征是将先验知识加入强化学习系统的一个重要方式。

### 多项式基和傅里叶基

![156](https://pic.imgdb.cn/item/60fbc6855132923bf8b96b68.png)

另一类线性函数逼近方法基于历史悠久的傅里叶级数，它会将周期函数表示为正弦和余弦基函数的加权和。

The usual Fourier series representation of a function of one dimension having period $\tau$ represents the function as a linear combination of sine and cosine functions that are each periodic with periods that evenly divide $\tau$ (in other words, whose frequencies are integer multiples of a fundamental frequency $1/\tau$ ).  如果你对有界区间内定义的非周期函数的近似感兴趣，也可以使用这些傅里叶基函数，并将$\tau$设置为区间长度。那么感兴趣的函数就是正弦和余弦函数的周期性线性组合的一个周期。

此外，如果将$\tau$设置为感兴趣区间长度的两倍，并将注意力集中在版区间$[0,\tau/2]$上的近似值，则可以只是用余弦函数。也可以只使用正弦函数。It is generally better to keep just the cosine features because “half-even” functions tend to be easier to approximate than “half-odd” functions because the latter are often discontinuous at the origin. 

![157](https://pic.imgdb.cn/item/60fbc6855132923bf8b96b79.png)

![158](https://pic.imgdb.cn/item/60fbc6855132923bf8b96b96.png)

傅里叶特征在不连续性方面存在问题，除非包含非常高频的基函数，否则很难避免不连续点周围的“波动”问题。由于傅里叶特征在整个状态空间上几乎是非零的（仅包含可忽略的零点），所以可以说它们表征了状态的全局特性，然而这却使得表征局部特性变得比较困难。

![159](https://pic.imgdb.cn/item/60fbc6855132923bf8b96baf.png)

### 粗编码（Coarse Coding）

考虑状态集在一个连续的二维空间上。一种表征的方式是将特征表示为下图所示的状态空间中的圆。If the state is inside a circle, then the corresponding feature has the value 1 and is said to be present; otherwise the feature is 0 and is said to be absent. 给定一个状态，其中二值特征表示该状态位于哪个圆内，因此就可以粗略地对其位置进行编码。这种表示状态重叠性质的特征（不一定是圆或者二值）被称为**粗编码**。

![160](https://pic.imgdb.cn/item/60fbc6855132923bf8b96bd6.png)

如果我们训练空间中的一个点，那么所有与这个状态相交的圆的权值将受到影响，这就产生了”泛化“。一个状态点所有”共有“的圆越多，其影响就越大。

![161](https://pic.imgdb.cn/item/60fbc6c45132923bf8ba72d2.png)

具有更大感受野的特征可以带来更强的泛化能力，但似乎这也会限制学到的函数的表达能力，使其成为一个粗近似而不能做出比感受野的宽度更细致的判别。Happily, this is not the case. **<font color=red>初始时从一个点推广到其他点的泛化能力确实是由感受野的大小与形状控制的，但最终可以达到的精细判别能力（敏锐度），则更多地是由特征的数量控制的</font>**。

![162](https://pic.imgdb.cn/item/60fbc6c45132923bf8ba72ee.png)

**<font color=red>在粗编码中，每一个状态中的值等于该状态激活的所有互相重叠的感受野算出的值的和，在上面这个例子中，每个点的value等于它所在的所有区间对应的$w$的和。</font>**

### 瓦片编码（Tile Coding）

瓦片编码是一种用于多维连续空间的粗编码。It may be the most practical feature representation for modern sequential digital computers.

在瓦片编码中，感受野组成了状态空间中的划分。每个划分被称为一个覆盖（tiling），划分中的每一个元素被称为一个瓦片（tile）。仅使用一次覆盖，我们得到的并不是粗编码而只是状态聚合的一个特例。

![163](https://pic.imgdb.cn/item/60fbc6c45132923bf8ba7306.png)

为了拥有粗编码的优点，我们需要使用多个覆盖，每一个动相对于瓦片宽度进行一定比例的偏移。

![164](https://pic.imgdb.cn/item/60fbc6c45132923bf8ba731f.png)

An immediate practical advantage of tile coding is that, because it works with partitions, the overall number of features that are active at one time is the same for any state. Exactly one feature is present in each tiling, so the total number of features present is always the same as the number of tilings. 这就使得步长$\alpha$​​​​​​​​​的取值简单而且符合直觉。例如，可以选$\alpha=\frac{1}{n}$​​​​​​，其中$n$​​​​​​是覆盖个数，这就是单次尝试（one-trial）学习。如果在样本$s \to v$​​​​​​上训练，那么无论先验估计 $\hat{v}(s, \mathbf{w_t})$​​​​​​是多少，新的估计总是 $\hat{v}(s,\mathbf{w_{t+1}})=v$​​​​​​.（<font color=red>这里我没弄懂是为什么</font>）通常为了使得目标输出具有较好的泛化性以及一定的随机波动，我们会希望更新的变化比这个更慢一些。例如$\alpha=\frac{1}{10n}$​​​​​​​​​.

![165](https://pic.imgdb.cn/item/60fbc6c45132923bf8ba733c.png)

在所有例子中，不同覆盖之间的偏移量在每个维度上都与瓦片宽度成比例。如果用$w$来表示瓦片的宽度，$n$来表示覆盖的数量，那么$\frac{w}{n}$是一个基本单位。在一个边长为$\frac{w}{n}$的小方块内，所有的状态都会激活同样的瓦片。在二维空间中，如果我们说每个覆盖都以位移量（1,1）偏移，则意思是每个覆盖相对于前一个覆盖的偏移量是$\frac{w}{n}$乘上这个向量。上图下半部分的非对称偏移覆盖是以（1,3）偏移。

![166](https://pic.imgdb.cn/item/60fbc7445132923bf8bc8c95.png)

在选择覆盖策略时，我们需要挑选覆盖的个数以及瓦片的形状。The number of tilings, along with the size of the tiles, determines the resolution or fineness of the asymptotic approximation。<font color=red>在一个维度上被拉长的瓦片，会促进这个方向上的泛化</font>。

在实际运用中，最好在不同的覆盖中使用不同形状的瓦片。

![167](https://pic.imgdb.cn/item/60fbc7445132923bf8bc8ca0.png)

### 径向基函数

径向基函数（Radial basis functions， RBF）是粗编码在连续特征中的自然推广。每个特征不再是0或者1，而可以是区间【0,1】中的任何值，这个值反映了“出席”的程度。

![168](https://pic.imgdb.cn/item/60fbc7445132923bf8bc8cb6.png)

RBF相对于二值特征的主要优点在于它们可以生成光滑且可微的近似函数。虽然这很吸引人，但其实在大多数情况下这并没有实际的意义。Nevertheless, extensive studies have been made of graded response functions such as RBFs in the context of tile coding (An, 1991; Miller et al., 1991; An et al., 1991; Lane, Handelman and Gelfand, 1992). All of these methods require substantial additional computational complexity (over tile coding) and often reduce performance when there are more than two state dimensions. In high dimensions the edges of tiles are much more important, and it has proven difficult to obtain well controlled graded tile activations near the edges.

## 手动选择步长参数

随机近似的理论给出了缓慢减小的步长序列上的条件（求和等于无穷，平方求和小于无穷），可以保证收敛，然而会使得学习变得很慢。在表格型MC方法中，产生随机样本的经典选择$\alpha_t=1/t$，不适用于TD方法，不适用于非平稳问题，也不适用于任何函数逼近。

为了更直观地感受应该如何手动设置步长，我们先回到表格型的情况。在此种情况下，我们可以理解步长$\alpha=1$​会导致更新后的样本误差被完全抹除。但通常情况下，我们希望学习比这要慢一些。如果$\alpha=1/\tau$​，那么一个状态的表格型估计将在大约$\tau$​次经验之后接近它的目标均值，且更近的目标影响更大。

**对于一般的函数逼近，则没有一个清晰的“状态的经验数量”的概念，因为每一个状态都可能与其他状态存在某种程度上的相似性或步相似性。**然而，在线性函数逼近的情况下，假设你希望从基本上相同的特征向量的$\tau$次经验来学习，一个好的粗略经验法则是

![169](https://pic.imgdb.cn/item/60fbc7445132923bf8bc8ce0.png)

这种方法在特征向量的长度变化不大时特别有效。在理想情况下，$\mathbf{x}^T\mathbf{x}$为常数。

## 最小二乘时序差分

目前为止讨论的所有方法都需要在每一个时刻进行正比于参数个数的计算。然而，通过更多的计算其实可以做得更好。

![170](https://pic.imgdb.cn/item/60fbc7445132923bf8bc8cff.png)

可能有人就会问了，为什么必须要迭代地求解呢？这是对数据的一种浪费，我们能否通过对$\mathbf{A,b}$的估计，然后直接计算TD的不动点呢？Least Square TD就是这样做的。

![171](https://pic.imgdb.cn/item/60fbc79f5132923bf8be0eb4.png)

上式是线性TD(0)算法的数据效率最高的一种形式，但是计算复杂度也更高。半梯度TD(0)要求的内存和每步的计算复杂度都只有$O(d)$。LSTD的复杂度是多少呢？复杂度似乎会随着t的增加而增加，但是$\hat{A}_t,\hat{b}_t$可以使用增量式的算法实现，所以每一步的计算时间可以保持在常数复杂度内。即使如此，$\hat{A}_t$的更新会涉及到外积计算，它的计算复杂度是$O(d^2)$，当然它的空间复杂度也是$O(d^2)$。

一个潜在的更大的问题是要计算$\hat{A}_t$的逆，计算复杂度是$O(d^3)$。幸运的是，在这个特殊的式子（外积的和），矩阵求逆运算可以采用复杂度为$O(d^2)$的增量式更新。

![172](https://pic.imgdb.cn/item/60fbc79f5132923bf8be0ee2.png)

而$\hat{A}_0=\epsilon I$。尽管被称为Sherman-Morrison公式的这个式子看上去很复杂，但它只涉及到“向量-矩阵”和“向量-向量”的乘法，因此计算复杂度为$O(d^2)$。

Whether the greater data efficiency of LSTD is worth this computational expense depends on how large d is, how important it is to learn quickly, and the expense of other parts of the system. 虽然LSTD经常被宣扬具有不需要设置步长参数的优势，但这个优势可能被夸大了。虽然LSTD不需要设置步长，但是需要参数$\epsilon$，如果$\epsilon$设置得太小，则这些逆可能变化非常大，如果$\epsilon$设置得过大，则学习过程会变得很慢。另外，**LSTD没有步长意味着它没有遗忘机制**。虽然有时候这是我们想要的，但是当目标策略像在GPI中那样不断变化时，这就会产生问题。在控制问题中，LSTD通常和其他有遗忘机制的方法相结合，这使得它不需要设置步长的优势消失了。

### 基于记忆的函数逼近

仅仅在记忆中保存看到过的训练样本（至少保存样本的子集）而不更新任何参数。每当需要知道查询状态的价值估计值时，就从记忆中检索查找出一组样本，然后使用这些样本来计算查询状态的估计值。这种方法优势被称为拖延学习（lazy learning），因为处理训练样本被推迟到了系统被查询的时候。取了个看上去很高大上的名字，其实本质就是非参数估计......。

![173](https://pic.imgdb.cn/item/60fbc79f5132923bf8be0ef7.png)

Being nonparametric, memory-based methods have the advantage over parametric methods of not limiting approximations to pre-specified functional forms.  基于记忆的局部近似法还有其他的性质，使得它非常适合用于强化学习。它可以将函数逼近集中在真实或者模拟轨迹中的访问过的状态（“状态-动作”）的局部邻域。全局近似估计可能是没有必要的，因为状态空间中的许多区域永远不会（或者几乎不会）被访问到。

避免全局估计也是解决维度灾难的一种方法。当然，遗留下来的最关键的问题，就是基于记忆的方法能否足够快地对回复智能体有用的查询。A related concern is how speed degrades as the size of the memory grows. Finding nearest neighbors in a large database can take too long to be practical in many applications.

Proponents of memory-based methods have developed ways to accelerate the nearest neighbor search.  一种方法是使用特殊的多维数据结构来储存训练数据。比如k-d树（k维树的简称），它递归地将k维空间划分成二叉树节点可以表示的空间。依据数量和数据在状态空间中的分布方式，使用k-d树可以在近邻搜索中很快排除空间中的大片无关区域，使得对某些暴力搜索会花费很长时间的问题的搜索变得可行。

## 基于核函数的函数逼近

基于记忆的方法，如加权平均法，会给数据库中的样本$s' \to g$分配权值。而权值往往基于$s'$和查询状态$s$之间的距离。分配权值的函数我们称之为核函数。更一般的，权值不是一定取决于距离，也可以基于一些其他衡量状态间相似度的方法。

从稍微不同的角度看，可以认为$k(s,s')$是对从$s'$到$s$的泛化能力的度量。核函数将任意状态对于其他状态的相关性知识数值化地表示了出来。

核函数回归是一种基于记忆的方法。

![174](https://pic.imgdb.cn/item/60fbc79f5132923bf8be0f0f.png)

任何线性参数化回归方法，状态有特征向量$\mathbf{x}(s)=(x_1(s),x_2(s),\ldots,x_d(s))^T$表示的，都可以被重塑为核函数回归，在这里的核函数$k(s,s')=\mathbf{x}(s)^T\mathbf{x}(s')$.

相比于为了线性参数化函数逼近模型构造特征，我们可以直接构造特征向量的核函数。不是所有的核函数都可以表示成特征向量的內积，但是可以这样表示的核函数对于等价的参数化方法有着显著的优势。上式有一个简洁函数形式来避免涉及d维空间中內积的计算，它支持在巨大高维特征空间中高效地工作，而实际上只需要处理记忆中存储的样本。

## 深入了解同轨策略学习：“兴趣”与“强调”（Interest and Emphasis）
目前为止，我们对遇到的所有状态都是平等地看待。然而，在某些场景中，我们会对有些状态会比对其他的状态更感兴趣。In discounted episodic problems, for example, we may be more interested in accurately valuing early states in the episode than in later states where discounting may have made the rewards much less important to the value of the start state. Or, if an action-value function is being learned, it may be less important to accurately value poor actions whose value is much less than the greedy action.

我们平等对待所有状态的原因是我们根据同轨策略分布进行更新，在这种情况下半梯度方法可以得到更好的理论结果。其实，对于一个MDP，我们可以有很多同轨策略分布。所有的分布都遵从从目标策略运行时在轨迹中遇到的状态分布，但是它们在不同的轨迹中是不同的。从某种意义上说，<font color=red>就是源于轨迹的初始化不同</font>。

接下来介绍一些新概念。首先介绍一个非负的随机标量变量$I_t$，称之为兴趣值。表示我们在t时刻有多大的兴趣要精确估计一个状态（或“状态-动作”）的价值。If we don’t care at all about the state, then the interest should be zero; if we fully care, it might be one, though it is formally allowed to take any non-negative value.  兴趣值可以用任何有前后因果关系的方式来设置，例如，可以基于从初始时刻到t时刻的部分轨迹，或者基于在t时刻学到的参数。在$\overline{VE}$中的分布$\mu$，现在被定义为遵从目标策略运行过程中所遇到的状态的疫兴趣值为权值的加权分布。其次，我们介绍另一个非负的随机标量变量，即强调值$M_t$。这个标量会被乘上学习过程中的更新量，因此决定了在t时刻强调或不强调“学习”。给定这些定义，我们可以用更一般的n步学习法则来进行更新，

![175](https://pic.imgdb.cn/item/60fbc79f5132923bf8be0f27.png)

下面这个例子展示了兴趣值和强调值如何能够更精确地估计价值函数。

![176](https://pic.imgdb.cn/item/60fbc7e25132923bf8bf20db.png)

线性半梯度n步TD已被证明在标准条件下对于所有的n都收敛，并且会收敛到最优误差的一个边界$\overline{VE}$以内（可由蒙特卡洛法渐近地获得）。这个边界总是随着n的增大而变小，并且在$n \to \infty$趋于0. 然而，在实际中，选择很大的n会导致非常慢的学习。中等程度的自举法通常是首选。

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。
