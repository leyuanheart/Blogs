---
layout:     post
title:      Reinforcement Learning-An Introduction Chapter5
subtitle:   Monte Carlo Methods
date:       2021-04-20
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Reinforcement Learning
---

<img src="https://pic.imgdb.cn/item/60fbafce5132923bf857fcd4.jpg" alt="RL" style="zoom:20%;" />

之前介绍的赌博机算法是采样并平均每个动作的收益（reward），蒙特卡洛算法与之类似，采样并平均每一个“状态-动作”二元组的回报（return）。这里主要的区别是：<font color=red> 我们现在有多个状态，每一个状态都类似于一个不同的赌博机问题。并且这些问题是相互关联的</font>。

## 蒙特卡洛预测

在给定的某一幕中，每次状态$s$的出现都称为对$s$的一次访问（visit）。在同一幕中，$s$可能会被多次访问到。在这种情况下，我们称第一次访问为$s$的首次访问（first visit）。

因此蒙特卡洛算法也分成两种：

- 首次访问型蒙特卡洛（first-visit MC）
- 每次访问型蒙特卡洛（every-visit MC）

These two Monte Carlo (MC) methods are very similar but have slightly different theoretical properties.

首次访问型MC，其收敛速度是$\frac{1}{\sqrt{n}}$，$n$表示被平均的回报值个数；而每次访问型MC中，它会二阶收敛到$v_{\pi}(s)$（这个要去看论文[Singh and Sutton, 1996](https://link.springer.com/content/pdf/10.1007/BF00114726.pdf)）。

“首次访问型蒙特卡洛”比较常用，而“每次访问型蒙特卡洛”则能更自然地扩展到函数近似与资格迹（eligibility trace）方法。

### 二十一点

![28](https://pic.imgdb.cn/item/60fbb4275132923bf86a5ae0.png)

这个例子里，玩家的策略是：在手牌点数之和小于20时要牌，否则停牌。图中上下两行分别是玩家有无*可用*A时计算的不同状态下的价值函数。

分析一下这个图，有几点特点：

1. 都是在最后两行价值函数突然增高。原因是，玩家的策略就是在小于20时要牌，而要牌就有爆的可能性，所以价值函数会比较低，但是在20和21处，玩家会停牌，而却庄家的固定策略是在手牌数超过17就停牌，所以玩家获胜的概率很大，因此价值很高。
2. 虽然最后两行是增高的，但是在最靠左侧的一列却降低了。原因是最左侧是庄家露出的牌是A，A是比较特殊的牌，它既可以当1，也可以当11，所以当庄家有A时，他获胜的纪律会增大一些，反之就会减少玩家在此处的价值。
3. 图上面一行中靠前的值比下面一行对应的位置要高。原因仍然是“特殊的A”，当有“特殊的A”时，玩家获胜机会就会大一些。

#### 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。

An important fact about Monte Carlo methods is that <font color=red> 对每个状态的估计是独立的</font>。它对于一个状态的估计完全不依赖于对其他状态的估计，这与动态规划（DP）完全不同。

特别需要注意的是：对于蒙特卡洛算法来说，**计算一个状态的代价与状态的个数是无关的**。这使得<font color=red>MC算法适合在仅仅需要获得一个或者一个子集的状态的价值函数时使用</font>。我们可以从这个特定的状态开始采样生成一些幕序列，然后获取回报的平均值，而完全不需要考虑其他的状态。这是MC相比于DP的第三个优势（另外两个是：可以从实际经历和模拟经历中学习。不过我觉得这也不一定就算优势了）。

## 动作价值的蒙特卡洛估计

With a model, state values alone are sufficient to determine a policy: one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state. 但在没有模型的情况下，仅仅有状态价值函数是不够的。我们必须通过显示地确定每一个动作的价值函数来确定一个策略。

<font color=red>这里唯一的复杂之处在于一些“状态-动作”二元组可能永远不会被访问到</font>。如果$\pi$是一个确定性（deterministic）的策略，那么遵循$\pi$意味着在每一个状态中只会观测到一个动作的回报。在无法获取回报进行平均的情况下，MC算法无法根据经验改善动作价值函数的估计。这个问题很严重，因为学习动作价值函数就是为了帮助在每个状态的所有可用的动作之间进行选择。To compare alternatives we need to estimate the value of all the actions from each state, not just the one we currently favor.

这其实就是一个如何保持**试探**的问题。一种方法是将指定的“状态-动作”二元组作为起点开始一幕采样，同时保证所有“状态-动作”二元组都有非零的概率可以被选为起点。我们把这种假设称为**试探性出发**（exploring starts）。

试探性出发假设有时非常有效，但是也并非总是那么可靠。当直接从真实环境中进行学习时，这个假设就很难满足了。作为替代，另一种常见的确保每一个“状态-动作”二元组都被访问到的方法是：**只考虑那些在每个状态下所有动作都有非零概率被选中的随机策略**。

## 蒙特卡洛控制

这里就沿用广义策略迭代（GPI）的思想。从初始策略开始，不断地进行策略评估和策略改进，直到收敛。策略改进的过程和之前之前policy iteration一样，就是对当前的价值函数进行贪心处理。策略评估则有之前的动态规划变成上面提到的蒙特卡洛预测。

这个算法最终能收敛的条件是蒙特卡洛算法的收敛性。而这需要满足两个很强的假设：

1. 试探性出发（exploring starts）
2. 进行策略评估时有无限多的样本序列进行试探

但是为了得到一个实际可用的算法，我们必须去除这两个假设。首先考虑无限多样本序列这一个。这个其实比较好去除。事实上，在经典的DP算法中，策略评估过程也仅仅是渐近地收敛到真实的价值函数（迭代次数不可能是无穷）。有两个方法可以解决这一问题。

一种方法是<font color=red>定量地去评估对$q_{\pi_k}$估计的好坏</font>。这就需要做出一些假设并定义一些测度，来分析逼近误差的幅度和出现概率的上下界，然后采取足够多的步数来保证这些界足够小。This approach can probably be made completely satisfactory in the sense of guaranteeing correct convergence up to some level of approximation. However, it is also likely to require far too many episodes to be useful in practice on any but the smallest problems.

第二种方法是不再要求在策略改进前就完成策略评估。即在每一个评估步骤中，我们让动作价值函数往$q_{\pi_k}$方向逼近，但是我们并不要求它要非常接近真实的值。一种极端的形式就是价值迭代，在相邻的两步策略改进中只进行一次策略评估的迭代。

因此，对于蒙特卡洛策略迭代（MCPI），也可以逐幕交替进行评估与改进。每一幕结束后，使用观测到的回报进行策略评估，然后在该幕序列访问到的每一个状态上进行策略的改进。这个算法叫做**基于试探性出发的蒙特卡洛（MCES）**。

![29](https://pic.imgdb.cn/item/60fbb4275132923bf86a5b01.png)

在MCES中，无论之前遵循的是哪个策略，对于每一个“状态-动作”二元组的所有回报都被累加并平均。可以很容易发现，MCES不会收敛到任何一个次优的函数。If it did, then the value function would eventually converge to the value function for that policy, and that in turn would cause the policy to change. 因为动作价值函数的变化随着时间增加而不断变小，所以应该一定能收敛到最优价值函数这个不动点，<font color=red>然而这一点还没有得到严格的证明</font>（部分解法可以参见[Tsitsiklis, 2002](https://www.mit.edu/~jnt/Papers/J089-02-jnt-optimistic.pdf)）。

### 解决二十一点问题

二十一点问题可以使用MCES，由于是模拟游戏，所以可以很容易地使所有可能的起点概率都不为零，确保试探性出发假设成立。在这个例子中，只需简单地随机等概率选择庄家的扑克牌、玩家手牌的点数，以及确定是否有可用的A即可。

![30](https://pic.imgdb.cn/item/60fbb4275132923bf86a5b16.png)

## 没有试探性出发假设的蒙特卡洛控制

如何避免这个非常难被满足的假设呢？唯一的一般性解决方案就是<font color=red>智能体能够持续不断地选择所有可能的动作</font>。有两种方法可以保证这一点，分别被称为**同轨策略**（on-policy）和**离轨策略**（off-policy）方法。On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas o↵-policy methods evaluate or improve
a policy different from that used to generate the data.

之前的MCES就是同轨策略方法的一个例子。接下来我们将要设计一种同轨策略的蒙特卡洛控制算法，使其不再依赖于不合实际的试探性假设。

在同轨策略中，策略一般是“软性”（soft）的，即对于任意的$s \in S$​以及$a \in A(s)$​，都有$\pi(a\|s) > 0$​，但它们会逐渐逼近一个确定性的策略。

所谓的$\epsilon$​​-软性策略是指，对于所有的状态和动作都有$\pi(a\|s) > \frac{\epsilon}{\|A\|}$​​。而我们要使用的同轨策略方法称为$\epsilon$​​-贪心测策略。

![31](https://pic.imgdb.cn/item/60fbb6d55132923bf875c6a3.png)

由于缺乏试探性出发假设，我们不能简单地通过对当前价值函数进行贪心优化来改进策略，否则就无法进一步试探非贪心动作。幸运的的是，<font color=red>GPI并不要求优化过程中所遵循的策略一定是贪心的</font>，只需要它逐渐逼近贪心策略即可。这就和在策略评估时一样，不需要其一定收敛，只要逐渐是往收敛方向去的就行。

虽然现在我们只能获得$\epsilon$-软性策略集合中的最优策略，但我们已经不再需要试探性出发假设了。Now we only achieve the best policy among the $\epsilon$-soft policies, but on the other hand, we have eliminated the assumption of exploring starts.

## 基于重要度采样的离轨策略

所有的学习控制方法都面临一个困境：它们希望学到的动作可以使随后的智能体行为是最优的，但是为了搜索所有的动作（以保证找到最优动作），它们需要采取非最优的心动。All learning control methods face a dilemma: They seek to learn action values conditional on subsequent optimal behavior, but they need to behave non-optimally in order to explore all actions (to find the optimal actions).

同轨策略方法实际上是一种妥协——它并不学习最优策略的动作值，而是学习一个接近最优而且仍能进行试探的策略的动作值。off-policy methods are often of greater variance and are slower to converge. On the other hand, off-policy methods are more powerful and general. 也可以将同轨策略方法视为一种目标策略与行为策略相同的离轨策略方法的特例。

我们首先从**预测**问题来讨论离轨策略方法。假设我们希望预测$v_{\pi}$或$q_{\pi}$，但是我们只遵循另外一个策略$b$（$b\neq \pi$）所得到的若干幕样本。在这种情况下，$\pi$是目标策略，$b$是行动策略，两个策略都固定且已知。

<font color=red>为了使用从$b$​​中得到的多幕样本序列去预测$\pi$​​，我们要求在$\pi$​​下发生的每个动作都至少偶尔能在$b$​​下发生</font>。换句话说，对于任意$\pi(a\|s)>0$​​，我们要求$b(a\|s)>0$​​。我们称其为**覆盖**（coverage）假设。It follows from coverage that $b$​​ must be stochastic in states where it is not identical to $\pi$​​. The target policy $\pi$​​, on the other hand, may be deterministic.

几乎所有的离轨策略方法都采用了**重要度采样**。

![32](https://pic.imgdb.cn/item/60fbb6d55132923bf875c6b7.png)

Thus, the relative probability of the trajectory under the target and behavior policies (the importance sampling ratio) is

![33](https://pic.imgdb.cn/item/60fbb6d55132923bf875c6b7.png)

最终，重要度采样比只与两个策略和样本序列数据相关，而与MDP的动态特性（状态转移概率）无关。

![34](https://pic.imgdb.cn/item/60fbb6d55132923bf875c6ec.png)

Now we are ready to give a Monte Carlo algorithm that averages returns from a batch of observed episodes following policy $b$ to estimate $v_{\pi}(s)$.

为了方便起见，我们在这里对时刻进行编号时，即使时刻跨越幕的边界，编号也增加。That is, if the first episode
of the batch ends in a terminal state at time 100, then the next episode begins at time $t = 101$。这样我们就可以使用唯一的编号来指代特定幕中的特定时刻。

对于每次访问型方法，我们可以定义所有访问过状态$s$的时刻的集合为$\mathcal{T}(s)$。

对于首次访问型，$\mathcal{T}(s)$只包括在幕内首次访问状态$s$的时刻。

![35](https://pic.imgdb.cn/item/60fbb6d55132923bf875c6fc.png)

### 普通重要度采样

![36](https://pic.imgdb.cn/item/60fbb6fc5132923bf8766cfc.png)

### 加权重要度采样

![37](https://pic.imgdb.cn/item/60fbb6fc5132923bf8766d18.png)

如果分母为零，则上式也定义为零。

普通重要度采样是无偏的，但是方差很大。Suppose the ratio were 10, indicating that the trajectory observed is ten times as likely under the target policy as under the behavior policy. In this case the ordinary importance-sampling estimate would be ten times the observed return. That is, it would be quite far from the observed return even though the
episode’s trajectory is considered very representative of the target policy.

普通重要度采样的方差一般是无界的，因为重要度采样比的方差是无界的。加权重要度采样是有偏的（偏差渐进收敛到零），在加权估计中任何回报的最大权值都是1。In fact, assuming bounded returns, the variance of the weighted importance-sampling estimator converges to zero even if the variance of the ratios themselves is infinite ([Precup, Sutton, and Dasgupta 2001](https://d1wqtxts1xzle7.cloudfront.net/42892117/Off-Policy_Temporal-Difference_Learning_20160221-15392-11htf9t.pdf?1456042216=&response-content-disposition=inline%3B+filename%3DOff_policy_temporal_difference_learning.pdf&Expires=1616425510&Signature=KqXEjzds9lYAeqFHVpeud2QrA4XgPZ-uRvtPrx68T24HiRTKnYUAJo4b21WYgjxCbZBdgI2hVITEJENwKm00r6-MbF~s~MM-9DzHAcVDEni2xKTYSsp7TPsIWLvMbv7e0VPirqUl3R7jpg61rFSD66ZELKvHaNCdjYP1TKwGsYsCRUrXFtl9DWYKxPDkKOasVbmRyRDM87xccLdVFoxnmZjwkw912~a3gkoxbeIOKBz7gXor-k8AgN05wAZLEPjH84BacC-N1yZ50Reb6QN3bot1Vtyc6NWlfat89jQlvk0JNEnRT~71KRahJalfdhAFVn9mh092brakQrNXJ1-x3g__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA))。

在实际应用中，人们偏好加权估计，因为它的方差经常很小。尽管如此，我们不会完全放弃普通重要度采样，因为它更容易扩展到基于函数逼近的近似方法。

## 对二十一点游戏中的状态值的离轨策略估计

we evaluated the state in which the dealer is showing a deuce, the sum of the player’s cards is 13, and the player has a usable ace (that is, the player holds an ace and a deuce, or equivalently three aces). The data was generated by starting in this state then choosing to hit or stick at random with equal probability (the behavior policy). The target policy was to stick only on a sum of 20 or 21. The value of this state under the target policy is approximately −0.27726（这是利用目标策略独立生成1一亿幕数据后对回报进行平均得到的）。

![38](https://pic.imgdb.cn/item/60fbb6fc5132923bf8766d2f.png)

两种算法的错误率最终都趋于0，但是加权重要度采样在开始时错误率明显较低，这也是实践中的典型现象。但是在初期，加权重要度采样的错误率会先上升然后下降。加权重要度采样需要一定数量的轨迹来减少偏差。

## 无穷方差

普通重要度采样的估计方差通常是无穷的，尤其当<font color=red>缩放过的回报值</font>具有无穷的方差时，其收敛性往往不尽如人意，而这种现象在<font color=red>带环</font>的序列轨迹中进行离轨策略学习时很容易产生。

There is only one nonterminal state s and two actions, right and left. The right action causes a deterministic transition to termination, whereas the left action transitions, with probability 0.9, back to s or, with probability 0.1, on to termination. The rewards are +1 on the latter transition and otherwise zero. Consider the target policy that always selects left. All episodes under this policy consist of some number (possibly zero) of transitions back to s followed by termination with a reward and return of +1. Thus the value of s under the target policy is 1 ($\gamma$ = 1). Suppose we are estimating this value from off-policy data using the behavior policy that selects right and left with equal probability.

![39](https://pic.imgdb.cn/item/60fbb6fc5132923bf8766d47.png)

上图展示了基于普通重要度采样的估计的首次访问型MC算法的10次独立运行的结果。即使在数百万幕之后，预测值仍然没有收敛到正确的值 1。作为对比，加权重要度采样算法在<font color=red>第一次以向左的动作结束的幕之后就给出了 1的预测</font>。

想想是为什么？因为在这个特殊的例子里，如果行为策略（左右随机）与目标策略（一直向左）不一致，那么重要度采样比的值就是0。那么这些回报值对于加权重要度采样计算公式的分子和分母都没有贡献，即**只对与目标策略一致的回报进行加权平均**。

We can verify that the variance of the importance-sampling-scaled returns is infinite in this example by a simple calculation. 


$$
Var[X]=E[(X-E[X])^2]=E[X^2]-(E[X])^2
$$


如果假设回报的期望是有界的，那么当且仅当随机变量的平方的期望是无穷时，方差才为无穷。因此我们需要证明下式是无穷：

![40](https://pic.imgdb.cn/item/60fbb6fc5132923bf8766d6d.png)

我们把它根据幕长度和终止情况进行分类。首先，所有以向右的动作结束的幕，都不会对该式产生影响，因此可以忽略。其次，所有幕的回报值都是1，所以$G_0$也可以省略不写。所以，我们只需要考虑每种长度的幕，将每种长度的幕出现概率乘以重要度采样比的平方，再将它们加起来：

![41](https://pic.imgdb.cn/item/60fbb7345132923bf8775dde.png)

**问题：如果采用每次访问型MC方法，估计器的方差仍然会是无穷吗？**

对于每次访问型MC方法，要评估的目标变成了：


$$
E_b \left[ \sum_{k=1}^{T-1} \left(\prod_{t=0}^{k}\frac{\pi(A_t|S_t )}{b(A_t|S_t)}G_0\right)^2\right]
$$


这是针对一条轨迹来说，可以看到其实本质上和首次访问型没有区别，只是需要求和的项变多了。需要关注的仍然是小括号中的项期望。

<font color=red>遗留问题：当使用加权重要度采样时，证明经过加权缩放后的方差是有界的。</font>

## 增量式实现

对于普通重要度采样，增量式的实现就是之前介绍过的incremental mean，但是对于加权重要度采样，需要稍微进行一些修改。

我们希望得到的估计是：


$$
V_n=\frac{\sum_{k=1}^nW_kG_k}{\sum_{k=1}^nW_k}, n \geq 1
$$


为了能不断跟踪$V_n$​的变化，必须为每一个状态维护前$n$​个回报对应的权值的累加和$C_n$​。$V_n$​​的更新方法是：


$$
V_{n+1}=\frac{C_nV_n+W_{n+1}G_{n+1}}{C_n+W_{n+1}}=\frac{(C_nV_n+W_{n+1}V_n)+(W_{n+1}G_{n+1}-W_{n+1}V_n)}{C_n+W_{n+1}}=V_n+\frac{W_{n+1}}{C_n+W_{n+1}}(G_{n+1}-V_n)
$$


![42](https://pic.imgdb.cn/item/60fbb7345132923bf8775de9.png)

之前忘记提了，就是对于MC类的算法，都有一个特点：**就是更新是从轨迹的末尾开始的**。这其实是为了更有效率地计算回报。因为关于$G_t$其实也是有一个迭代式的：$G_{t}=R_{t+1}+\gamma G_{t+1}$。而我们是知道轨迹结束时刻的回报的（就是等于结束时的收益（reward）），所我们可以从后向前进行迭代。

## 离轨策略蒙特卡洛控制

Recall that the distinguishing feature of on-policy methods is that they estimate the value of a policy while using it for control. In off-policy methods these two functions are separated. The policy used to generate behavior, called the **behavior policy**, may in fact be unrelated to the policy that is evaluated and improved, called the **target policy**. An advantage of this separation is that the target policy may be deterministic (e.g., greedy), while the behavior policy can
continue to sample all possible actions.

<font color=red>这些方法要求行动策略对目标策略可能做出的所有动作都有非零的概率被选中。</font>

![43](https://pic.imgdb.cn/item/60fbb7345132923bf8775e0a.png)

解释一下这个算法。首先target policy $\pi$​​​是一个贪心策略，behavior policy是任意的一个soft policy。然后在每一条轨迹内的更新中，如果$\arg \max_a Q(S_t, a) \neq A_t$​​​，那么就意味着$\pi(A_t\|S_t)=0$​​​，则采样比$\rho_{t+1:T-1}$​​​就为0了，$t$​​​时刻以前的$G_t$​​​都没有作用了，所以就结束掉这条轨迹的计算，跳转到下一条轨迹（也就是说从轨迹末尾开始更新，直到更新到$\arg \max_a Q(S_t, a) \neq A_t$​​​这个时间点$t$​​​结束），反之，如果$\arg \max_a Q(S_t, a) = A_t$​​​，则意味着$\pi(A_t\|S_t)=1$​​​，采样比就变成了$\frac{1}{b(A_t\|S_t)}$​​​。

这个算法有一个潜在的问题是：如果轨迹的后半部分出现太多非贪心的侧率，即行为策略的$A_t$总是不等于$\arg \max_a Q(S_t, a)$，那么学习速度会很慢。There has been insufficient experience with off-policy Monte Carlo methods to assess how serious this problem is. If it is serious, the most important way to address it is probably by incorporating **temporal difference learning**.

### 赛道问题

![44](https://pic.imgdb.cn/item/60fbb7345132923bf8775e21.png)

## 折扣敏感的重要度采样

当我们考虑折扣的时候，我们可以利用回报是每个时刻折后收益之和这一内部结构来大幅减少离轨策略估计的方差。

举个例子。假设$\gamma=0$​，对于一个持续100步的幕来说，0时刻的回报$G_0=R_1$​，但是它的重要度采样比却会是100个因子之积，即$\frac{\pi(A_0\|S_0)}{b(A_0\|S_0)}\frac{\pi(A_1\|S_1)}{b(A_1\|S_1)}\ldots\frac{\pi(A_99\|S_99)}{b(A_99\|S_9)}$​。但实际上只需要考虑第一个因子，后面99个都是无关的，因为在得到首次收益后，整幕的回报就已经决定了。These later factors are all independent of the return and of expected value 1; they do not change the expected update, but they add enormously to its variance. 在某些情况下，它们甚至可以使方差变得无穷大。

这个思路的本质是<font color=red>把折扣看作是幕终止的概率，或者说，部分终止的程度</font>。For any $\gamma \in [0, 1)$, we can think of the return $G_0$ as partly terminating in one step, to the degree $1 − \gamma$, producing a return of just the first reward, $R_1$, and as partly terminating after two steps, to the degree $(1−\gamma)\gamma$, producing a return of $R_1 +R_2$, and so on. 这里的部分回报被称为**平价部分回报**（flat partial returns）：

![45](https://pic.imgdb.cn/item/60fbb7345132923bf8775e39.png)

其中“平价”表示没有折扣，“部分”表示这些回报不会一直延续到终止，而是在$h$出停止，$h$被称为**视界**（horizon），$T$是幕终止的时间。传统的全回报$G_t$可被看作是上述平价回报的总和：

![46](https://pic.imgdb.cn/item/60fbb7605132923bf8782078.png)

现在我们需要使用一种类似的截断的重要度采样比来缩放平价部分回报。由于$\bar{G}_{t:h}$只涉及到视界$h$为止的收益，因此我们只需要用到$h$为止的概率值。我们定义如下普通和加权的折扣敏感的重要度采样估计：

![47](https://pic.imgdb.cn/item/60fbb7605132923bf8782090.png)

当$\gamma=1$时，它们和之前的估计是一样的。

## 每次决策型（Per-decision）重要度采样

注意这里的每次决策型（per-decision）和之前提到的每次访问型（every-visit）是不同的概念。

There is one more way in which the structure of the return as a sum of rewards can be taken into account in o↵-policy importance sampling, a way that may be able to reduce variance even in the absence of discounting (that is, even if $\gamma = 1$).

在离轨策略估计器中，分子中求和计算的每一项本身也是一个求和式

![48](https://pic.imgdb.cn/item/60fbb7605132923bf87820a1.png)

其中每个子项是一个随机收益和一个随机重要度采样比的乘积。以第一个子项为例，它可以写成

![49](https://pic.imgdb.cn/item/60fbb7605132923bf87820b6.png)

可以发现，在所有这些因子中，只有第一项是与收益$R_{t+1}$相关的，其他的比率均为期望值为1的独立随机变量，因此求完期望后就只剩下

![50](https://pic.imgdb.cn/item/60fbb7605132923bf87820d9.png)

因此，对于第$k$项，我们有

![51](https://pic.imgdb.cn/item/60fbb7cc5132923bf879eb25.png)

这样一来，原来的期望就可以写成

![52](https://pic.imgdb.cn/item/60fbb7cc5132923bf879eb39.png)

我们把这种思想称为**每次决策型**重要度采样。借助这个思想，对于普通重要度采样估计，我们就找到了一种替代方法来进行计算，以减少估计的方差

![53](https://pic.imgdb.cn/item/60fbb7cc5132923bf879eb4e.png)

Is there a per-decision version of weighted importance sampling? This is less clear. So far, all the estimators that have been proposed for this that we know of are not consistent (that is, they do not converge to the true value with infinite data).

## 总结

The Monte Carlo methods presented in this chapter learn value functions and optimal policies from experience in the form of sample episodes. This gives them at least three kinds of advantages over DP methods. 

First, they can be used to learn optimal behavior directly from interaction with the environment, with no model of the environment’s dynamics. 

Second, they can be used with simulation or sample models. For surprisingly many applications it is easy to simulate sample episodes even though it is difficult to construct the kind of explicit model of transition probabilities required by DP methods. 

Third, it is easy and efficient to focus Monte Carlo methods on a small subset of the states. A region of special interest can be accurately evaluated without going to the expense of accurately evaluating the rest of the state set.

 A fourth advantage of Monte Carlo methods, which we discuss later in the book, is that they may be less harmed by violations of the Markov property. This is because they do not update their value estimates on the basis of the value estimates of successor states. In other words, it is because they do not bootstrap.

## 代码

参见[Reinforcement_Learning_An_Introduction](https://github.com/leyuanheart/Reinforcement_Learning_An_Introduction)。