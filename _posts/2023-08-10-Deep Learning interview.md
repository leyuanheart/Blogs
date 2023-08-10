---
layout:     post
title:      Summary of "Deep Learning Interviews"
subtitle:   
date:       2023-08-10
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Deep Learning
    - Interview Question
---

![](https://pic.imgdb.cn/item/64d4536d1ddac507ccdeaf24.jpg)

这篇文章是对《Deep Learning Interviews》的一个总结。正如这本书封页上所说的：这本书之所以很有价值是在于它搜集了许多人工智能主题中**完全解决的问题**。这就意味着你不用担心对于书中的问题你得出的答案是否是正确的这个问题。（当然前提是你认为他的答案是正确的）

这本书每章（第一章Introduction除外）都是由问题和答案两部分构成。我会按照本书的脉络整理我觉得很好的、有助于加深对概念理解的问题和答案，感兴趣的也可以去找原书籍进行阅读。另外，阅读本书需要有基本的微积分、概率论和数理统计的知识，比如期望和方差的定义和计算、一些常见的概率分布、逻辑回归等。

## Chapter 2—Logistic Regression

> 问题：什么是“odds of success”?
> 
> 答案：中文翻译是成功的几率，它表示的是成功的概率($p$)和失败的概率($1-p$)的比值，即$odds = \frac{p}{1-p}$。另一个经常使用的概念是log-odds，就是odds取个对数，通常也叫logit。

> 问题：对于logistics模型的输出，通过什么样的变换可以将其转化成一个概率分布？
> 
> 答案：通常有两种方法，一种是经典的normalization（每一项都除以所有项的和），另一种是加一个softmax层（如果是二分类就是sigmoid函数）。通过刻画因变量的概率分布是具有更多信息的表示方法，因为相比于得到最终的决策结果（hard-decision），这种soft-decision还包括了处理最优选择之外的其他选择的信息。

> 问题：逻辑回归模型其实是一类更广泛模型的一个子集，这类模型被称为广义线性模型（GLM），一个GLM有3个组成部分，对于二分类逻辑回归，它们分别是什么？
> 
> 答案：
> 
> 1. Random Component ($Y$)，
> 
> 2. System component ($X$)，
> 
> 3. link function: sigmoid函数 ($y=\frac{1}{1+e^{-x}}$).

![](https://pic.imgdb.cn/item/64d453831ddac507ccdedbbf.png)

> 问题![](https://pic.imgdb.cn/item/64d453831ddac507ccdedcdc.png)

该问题主要是考察对于relative risk和odds ratio的理解，以及关于方差的计算。

> 答案：
> 
> 1. explanatory var是癌症的种类，response var是癌症是否根除。
> 
> 2. relative risk是指一组的风险相对于另一组风险的比值：$\frac{p_1}{p_2}$；而odds ratio是指一组的odds相对于另一组odds的比值：$\frac{p_1/(1-p_1)}{p_2/(1-p_2)}$。
> 
> 3. 根据题目中的数据，计算odds ratio=$\frac{ac}{bd}=\frac{560\times36}{260\times69}=1.23745$；relative risk=$\frac{560/(560+260)}{69/(69+36)}=1.0392$。虽然都大于1，但是也都比较接近于1，所以不好判断是两个变量是否存在相关性。
> 
> 4. 这一问有点麻烦，假设我们是要计算odds ratio ($\theta=\frac{p_1/(1-p_1)}{p_2/(1-p_2)}$)的95%置信区间，关键要计算$\theta$的方差。$p_1$和$p_2$是不相关的，为了将它们分开，我们考虑log-odds ratio ($\log \theta$)：$var(\log \theta)=var[\log\frac{p_1}{1-p_1}-\log\frac{p_2}{1-p_2}]=var(\log\frac{p_1}{1-p_1})+var(\log\frac{p_2}{1-p_2})$。因此，本质上是要算$\log \frac{p}{1-p}$的方差。这里要用到delta method来计算近似方差。下面我用$\hat{p}$表示$p$的估计值，我们知道$var(\hat{p})=\frac{p(1-p)}{n}$，且$\hat{p}$的极限分布是正态分布，而现在有$g(p)=\log \frac{p}{1-p}$，由delta method， $var(g(\hat{p}))\approx[g'(p)]^2var(\hat{p})=(\frac{1}{(p(1-p)})^2\times\frac{p(1-p)}{n}=\frac{1}{np(1-p)}$。以$p_1$为例，$var(\log \frac{p_1}{1-p_1})=\frac{1}{(a+b)\frac{a}{a+b}\frac{b}{a+b}}=\frac{a+b}{ab}=\frac{1}{a}+\frac{1}{b}$。因此$var(\log \hat{\theta})=\frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}=0.21886$，则$\log \hat{\theta}$的95%的置信区间$0.213052\pm1.95\times0.21886=(0.6398298,-0.2137241)$。因此，$\hat{\theta}$的95%的置信区间为$(e^{0.210}, e^{0.647})=(0.810, 1.909)$。
> 
> 5. 置信区间包含1，说明真实的odds ratio并没有显著地不同于1，故没有足够证据表明癌症的根除和癌症的种类相关。

> 问题：一个取1概率为$p$的二分类变量的Entropy定义为：$H(p)\equiv -p\log p-(1-p)\log (1-p)$。
> 
> 1. 画出$H(p)$随$p$变化的图
> 
> 2. $H(p)$和logit函数的关系
> 
> 答案：
> 
> ![](https://pic.imgdb.cn/item/64d453831ddac507ccdedd0d.png)
> 
> $$
> \begin{align*}
H'(p)= & -(\log p+1)-(-\log(1-p)-1) \\
     = & \log (1-p)-\log p = \log \frac{1-p}{p} \\
     = & -logit(p)
\end{align*}
> $$

> python小知识：在python中，最大的浮点数大约是$1.797\times10^{308}$，其logarithm是$709.78$。

## Chapter 3—Probabilistic Programming & Bayesian Deep Learning

> 问题：$X \sim Binomial(n,p)$，即$P(X=k)=C_n^kp^k(1-p)^{n-k}, k=0,1,\ldots,n$。求$X$的期望和方差。
> 
> 答案：![](https://pic.imgdb.cn/item/64d453831ddac507ccdedd90.jpg)

> 问题：假设在概率空间$H$中有两个事件$A$和$B$，简析贝叶斯公式。
> 
> 答案：![](https://pic.imgdb.cn/item/64d453841ddac507ccdede36.png)

> 问题：
> 
> ![](https://pic.imgdb.cn/item/64d453cc1ddac507ccdf732a.png)
> 
> 这题说的是，二战期间德军使用的Enigma machine进行加密通讯，然后已知发送$X$和$Z$的概率分别为$\frac{2}{9}$和$\frac{7}{9}$。但是英国军队使用了一些方法来阻碍通信，比如可以以$\frac{1}{7}$的概率将发送的$X$给变成$Z$。那么现在问：如果德国军队接受到了$Z$，那么发送者实际上发送的是$Z$的概率是多少？
> 
> 答案：
> 
> ![](https://pic.imgdb.cn/item/64d453cd1ddac507ccdf7353.jpg)

> 问题：log-likelihood function的一阶导数被叫做什么？以及定义一下**Fisher Information**。
> 
> 答案：对数似然函数的一阶导被称为Fisher Score Function，即$score = \frac{\partial l(\theta)}{\partial \theta}$
> 
> Fisher information, is the term used to describe the expected value of the second derivatives (the curvature) of the log-likelihood function, and is defined by:
> 
> $$
> I(\theta)=-\mathbb{E} \Big[\frac{\partial^2 l(\theta)}{\partial^2 \theta} \Big]
> $$
> 
> 如果满足求导和积分可交换，且假设模型正确指定的情况下，有：
> 
> $$
> I(\theta)=\mathbb{E}\left[ (\frac{\partial l(\theta)}{\partial \theta})^2 \right] =Var(\frac{\partial l(\theta)}{\partial \theta})
> $$
> 
> 并且对于任意一个$\theta$无偏估计$\hat{\theta}$，有$Var(\hat{\theta}) \geq \frac{1}{I(\theta)}$。

> 问题：证明“the family of **beta distributions** is **conjugate to a binomial likelihood**.”
> 
> 这可能是贝叶斯统计里使用最为广泛的一个结论。
> 
> 答案：
> 
> ![](https://pic.imgdb.cn/item/64d453cd1ddac507ccdf736f.png)这提供了一种非常简单的贝叶斯估计伯努利分布参数的方法。以扔硬币为例，假设我们一开始认为硬币是均匀的，所以我们假设正面的朝上的概率$\gamma$服从一个$B(0.5,0.5)$的分布，然后我们不断进行实验，每一次实验过后我们都可以更新参数的后验分布。比如，我们观察到$h$次正面朝上，$t$次反面朝上，则$\gamma$的后验分布更新为$B(0.5+h,0.5+t)$。

## Chapter 4—Information Theory

According to Shannon’s theory, information and uncertainty are two sides of the same coin: the more uncertainty there is, the more information we gain by removing the uncertainty.

![](https://pic.imgdb.cn/item/64d453cd1ddac507ccdf739a.png)

注意：本章中所有的数值计算都是使用以2为底的对数$\log_2$，该logarithm产生比特（bit）单位，是信息论中常用的计量单位。

> 问题：写出香农著名的关于不确定性的公式。
> 
> 答案：
> 
> $H(P)=-\sum_{j=1}^{N}P_j \log_2 P_j~~~~(bits~per~symbol)$，$where~ P=(P_1,\ldots,P_N)^T$。

> 问题：列出Entropy $H(P)$满足的三个natural properties.
> 
> 答案：
> 
> ![](https://pic.imgdb.cn/item/64d453cd1ddac507ccdf73e3.png)

> 问题：解释一下什么是“**Shannon bit**”.
> 
> 答案：
> 
> ![](https://pic.imgdb.cn/item/64d454411ddac507cce06f7f.png)

> 问题：从信息论的角度解释一下什么是“surprise”。
> 
> 答案：这和事件发生的可能性有关。当发生概率很低的事件实际发生时我们会感到惊讶，它更新了我们的认知，给我们带来更多的信息。也就是说，如果一件事发生的先验概率是1，那么它发生所包含的信息量为0，相反的，如果一件事发生的概率仅为0.1，那么它发生所包含的信息量就非常大，因为它说明有一些我们认为不太可能发生情况是存在的。（Entropy 也可以被称为average surprise。）

> 问题：假设一段信号源以概率$P_a$传输一段给定的信息$a$。进一步假定该信息以二进制的形式被编码（bit string）。香农就是通过Entropy来表示一段信息被压缩成二进制字符串的平均长度：$H=-\sum_aP_a \log_2 P_a$。
> 
> 举个例子：假设一个均匀分布的像素（pixel）点（取值从0到255）要被输入进一个深度学习pipeline里，那么平均需要多少bits？
> 
> 这里$p_i=\frac{1}{256},~i=1,\ldots,256$，所以$H=-\sum_i^{256} \frac{1}{256}\log_2 \frac{1}{256}=8$，即传输一个均匀分布像素点需要8bits.

> 判断以下说法是否正确：任给3个随机变量$X,Y,Z$，且$Y=X+Z$，则$H(X,Y)=H(X,Z)$，其中$H(X,Y)$表示$X,Y$的联合分布函数对应的entropy。
> 
> 答案：正确。因为给定$X,Y$我们可以确定$Z$，而给定$X,Z$我们可以确定$Y$，这两者所包含的信息量是一样的，故$H(X,Y)=H(X,Z)$。

> 问题：写出两个离散分布$P$和 $Q$的Kullback-Leibler divergence公式，并且用bits的角度来给出一个KL-divergence的直觉上的解释。
> 
> 答案：
> 
> ![](https://pic.imgdb.cn/item/64d454411ddac507cce06f8c.png)
> 
> 一种解释是：本来要传输来自$P$分布的$X$，但是错误地用$Q$分布来编码所引入的**额外**的bits。因为你不知道真实的分布（最佳的编码方式），所以你必须要额外付出一些bits，这也比较合理。这也是KL-divergence又被称为relative entropy原因。
> 
> ![](https://pic.imgdb.cn/item/64d454411ddac507cce06fa0.png)

> 问题：写出两个变量$X$和$Y$的mutual information的公式。
> 
> 答案：![](https://pic.imgdb.cn/item/64d454411ddac507cce06fcb.png)
> 
> Mutual information通常用来度量两个变量之间的相关性，当$X \perp Y$时，$I(X,Y)=0$。

> 问题：列举出决策树中常用的3种确定划分节点的度量方法。
> 
> 答案：![](https://pic.imgdb.cn/item/64d454411ddac507cce06ffa.png)

> 问题：
> 
> ![](https://pic.imgdb.cn/item/64d454811ddac507cce0f026.png)
> 
> 这一个二类问题，题目内容太长我就不介绍了，主要是这个表，其中$\gamma$是因变量，有5个样本，3个正例2个反例。然后$\theta_1$和$\theta_2$是两个自变量，表里是其对应的取值。现在想要构造一个决策树模型来进行分类。简述该如何做。
> 
> 答案：决策树就是在当前节点根据某种规则（比如Information gain）选择一个变量将样本进行分类，然后在新生成的节点上重复以上操作直到达到指定的终止条件，叶节点上的样本就可以用来做预测。
> 
> Information gain指的是按照某一变量进行分类后熵的减少。以上述问题为例，初始的Entropy为$H(\gamma)=-(\frac{2}{5} \log_2 \frac{2}{5}+\frac{3}{5} \log_2 \frac{3}{5})\approx 0.97095$。如果我们选择变量$\theta_1$作为拆分变量，我们可以计算拆分后的Entropy：
> 
> $$
> \begin{align*}
H(\gamma|\theta_1)= & -\sum_{j \in{T,F}}P(\theta_1=j)\left [\sum_{i \in {+,-}}P(\gamma=i|\theta_1=j) \log_2 P(\gamma=i|\theta_1=j) \right ] \\
                  = & -\frac{4}{5}(\frac{3}{4} \log_2 \frac{3}{4} + \frac{1}{4} \log_2 \frac{1}{4})-\frac{1}{5}(0) \\
                  \approx & 0.8112
\end{align*}
> $$
> 
> 因此，information gain为$IG(\theta_1)=H(\gamma)-H(\gamma \vert \theta_1) \approx 0.32198$.

> 问题![](https://pic.imgdb.cn/item/64d454811ddac507cce0f054.jpg)

> 补充下列句子：The relative entropy $D(p\Vert q)$ is the measure of <u>**difference**</u> between two distributions. It can also be expressed like a measure of the <u>**inefficiency**</u> of assuming that the distribution is $q$ when the true distribution is $p$.
> 
> ![](https://pic.imgdb.cn/item/64d454821ddac507cce0f08f.jpg)

> 补充下列句子：Mutual information is a Shannon entropy-based measure of dependence between random variables. The mutual information between $X$ and $Z$ can be understood as the <u>**reduction**</u> of the uncertainty in $X$ given $Z$:
> 
> $$
> I(X,Z):=H(X)-H(X|Z)
> $$
> 
> where $H$ is the Shannon entropy, and $H(X \vert Z)$ is the conditional entropy of $X$ given $Z$：$H(X\vert Z)=-\sum_zP(Z=z) \left [\sum_x P(X=x\vert Z=z) \log_2 P(X=x\vert Z=z) \right]$

## Chapter 5—Deep Learning: Calculus，Algorithmic differentiation

![](https://pic.imgdb.cn/item/64d454821ddac507cce0f104.png)

sigmoid function: $\sigma(x)=\frac{1}{1+e^{-x}},~~~\sigma'(x)=\sigma(x)(1-\sigma(x))$.

hyperbolic tangent function: $tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=2\sigma(2x)-1$,

$tanh'(x)=1-[tanh(x)]^2$,$arctanh(x)=tanh^{-1}(x)=\frac{1}{2}[\ln(1+x)-\ln(1-x)]$,

$(arctanh(x))'=\frac{1}{1-x^2}$.

这一章基本都是关于微积分的内容，感兴趣的可以自己看一下。

## Chapter 6—Deep Learning: NN Ensembles

> ![](C:\Users\leyuan\Desktop\Deep-Learning-interview-Blog\Deep-Learning-interview\20.png)
> 
> 答案：Stacking. In stacking, we first (phase 0) predict using several base learners and then use a generalizer (phase 1) that learns on top of the base learners predictions.

> 关于有放回和无放回抽样的若干知识点：
> 
> 1. 两者都是**等概率**抽样模型。
> 
> 2. 无放回抽样中，两个样本之间是**负相关**的。
> 
> 3. 对于抽样出来的样本均值，无放回比有放回的方差小。
> 
> 假设有总体$X_1,X_2,\ldots,X_N$，均值为$\bar{X}$，方差为$\sigma^2(X)$。从中抽取$n$个样本$x_1,x_2,\ldots,x_n$。<font color=red>（以下分析注意大小写的区别）</font>
> 
> 首先对于有放回抽样，很好分析，因为每次抽样都是独立且同分布的，所以对于$\bar{x}=\frac{1}{n}\sum_i^nx_i$，$E[\bar{x}]=\frac{1}{n}\sum_i^n E[x_1]=\bar{X}$，$\sigma^2(\bar{x})=\frac{1}{n^2}\sum_i^n \sigma^2(x_1)=\frac{1}{n}\sigma^2(X)$。
> 
> 对于无放回抽样，每次抽样并不是独立的，所以计算会稍微复杂一些。首先是期望，$E[\bar{x}]=\frac{1}{n}\sum_i^n E[x_i]$。对于$E[x_1]$，显然$=\frac{1}{N}\sum_i^NX_i=\bar{X}$。
> 
> $E[x_2]$表示第二次抽样的样本期望，其公式可以写成$\sum_i^N P_iX_i$，其中$P_i$表示$x_2=X_i$的概率。因为是不放回抽样，所以这个概率应该等于第一次没有抽到$X_i$的同时第二次抽样$X_i$的概率，即$P_i=P(x_1 \neq X_i)P(x_2=X_i)=\frac{N-1}{N}\frac{1}{N-1}=\frac{1}{N}$，故$E[x_2]=\bar{X}$。同理，对于每个$i$，都有$E[x_i]=\bar{X}$。因此，$E[\bar{x}]=\bar{X}$。
> 
> 再考虑方差
> 
> $$
> \begin{gather*}
\sigma^2(\bar{x})=E\left[ (\bar{x}-E[\bar{x}])^2 \right]=E[(\bar{x}-\bar{X})^2] =E\left[((\frac{x_1+\ldots+x_n}{n}) - \bar{X})^2 \right]  \\
= \frac{1}{n^2} \Big[\sum_i^nE[(x_i-\bar{X})^2] + \sum_{i \neq j}E[(x_i-\bar{X})(x_j-\bar{X})]   \Big]
\end{gather*}
> $$
> 
> 大中括号里的第一个求和项里的每一项等于$\sigma^2(X)$，第二个求和项里一共有$n(n-1)$项，<font color=red>我们单独考虑其中的一项$E[(x_i-\bar{X})(x_j-\bar{X})]$</font>。
> 
> ![](https://pic.imgdb.cn/item/64d455091ddac507cce21b2f.jpg)
> 
> 注：<font color=blue>无放回抽样的方差只产生于没有抽出来的样本，因为抽出来的样本已经被观测到了，即如果$n=N$，则方差为0 </font>,<font color=red>但有放回抽样因为每次抽样是独立的，所以方差和$N$无关，就是$\sigma^2(X)/n$</font>。

Bagging is **variance** reduction scheme while boosting reduced **bias**.

> 列举几个在深度学习中常用的Ensembling方法。

1. Snapshot Ensembling

![](https://pic.imgdb.cn/item/64d455091ddac507cce21b86.jpg)

2. Random seed Ensembling

![](https://pic.imgdb.cn/item/64d455091ddac507cce21d5e.jpg)

3. Multi-model Ensembling

![](https://pic.imgdb.cn/item/64d455091ddac507cce21f96.jpg)

4. Learning rate schedules in Ensembling![](https://pic.imgdb.cn/item/64d455091ddac507cce22108.jpg)

## Chapter 7—Deep Learning: CNN Feature Extraction

> 问题：已知$1~bit = 0.000000125 MB$，则如果一个VGG-Net有138, 357, 544个浮点型参数，那么需要多大的存储空间来维持一个训练好的VGG-Net？
> 
> 答案：$138357544 \times 32 = 4427441408~bits = 553.430176 MB$。

这一章中有一节挺有意思的，是关于Neural style transfer。就是给定一张content image和一张style image，要求生成一张具有这种style的content图片。采用的方法是参考这篇文章[A neural algorithm of artistic style](https://arxiv.org/pdf/1508.06576.pdf)。它是用一个已经训练好的CNN模型作为Feature extraction，然后作用于content image和style image分别提取content feature和style feature，再将模型作用于一个随机噪声图片，将每个像素点作为参数，构造content loss和style loss来进行训练，使得更新后的图片兼具风格和内容。感兴趣的可以去看一下这篇文章。

## Chapter 8—Deep Learning

![](https://pic.imgdb.cn/item/64d455851ddac507cce35d36.png)

> 问题：以下公式经常出现在图像处理中：$(f *g)(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau$。请问这个公式代表什么？$g(t)$表示什么？如何理解这个式子？
> 
> 答案：这是卷积公式。在图像处理中$g(t)$表示filtering kernel。因为积分就是求和的极限，所以我们通过离散版的公式来理解。假设$f$和$g$的定义域都是$\{0, 1, 2, 3, 4, 5\}$。计算过程见下图：
> 
> ![](https://pic.imgdb.cn/item/64d455851ddac507cce35d68.jpg)
> 
> 可以看到，在计算$h(0)$时，$f$的位置保持不变，而$g$水平翻转了，然后是对应位置相乘再相加，由于定义域的限制，所以求和项里只有一项；而在计算$h(1)$时，$g$翻转后又向右平移了一个单位，求和项里就变成两项，分别是$f(0)$对应 $g(1)$和$f(1)$对应$g(0)$，再以此类推到$h(5)$。其实还可以继续往前推，$h(6),h(7)$，一直可以到$h(10)$。
> 
> 如果我们将这两个离散的函数看作是两个向量，再看$h(5)$的计算公式，会发现它和我们常见的內积公式$f^Tg$很像，唯一的区别就是将$g$翻转之后再和$f$做內积。而我们知道內积是衡量两个向量间的线性相关性，所以你会发现其实**卷积（convolution）操作和相关性（correlation）操作就只是差了一个翻转**。但其实差别不止这一个，还有一个是在计算相关性时，要使用$f$的共轭函数$f^*$。
> 
> ![](https://pic.imgdb.cn/item/64d455851ddac507cce35dac.png)

上述是一维的情况，在CNN中，我们处理的是图片，所以通常都是二维的（先不考虑RGB三通道），$f$通常就是我们图片，$g$通常表示卷积核。要注意的是，在实际应用的时候，我们并没有把卷积核进行水平和垂直翻转，而是直接对应图片位置上的值进行计算。所以，我们实际应用的所谓“卷积”操作，其实是求cross-correlation。

> Sigmoid函数的近似
> 
> ![](https://pic.imgdb.cn/item/64d455851ddac507cce35dee.jpg)

> Swish function
> 
> ![](https://pic.imgdb.cn/item/64d455851ddac507cce35eaa.png)

> Performance Metrics
> 
> ![](https://pic.imgdb.cn/item/64d455e61ddac507cce437cc.jpg)

> 问题：给定一个输入是$n\times n$维的，filter的size是$f \times f$，stride是 $s$，padding是$p$，问输出的维度是多少？
> 
> 答案：$\lfloor \frac{n-f+2p}{s} \rfloor+1$。

<font color=red>如果activation function在零点处的取值为0，那么Dropout在activation function之前和之后使用是等价的</font>。

虽然在Batch Normalization的original paper中是将BN层放在激活函数之前，即$CONV \to BN \to activation \to Dropout$。但是也有研究表明，将BN层放在激活函数之后会带来更好的效果，即$CONV \to activation \to BN \to Dropout$。

> 问题：
> 
> ![](https://pic.imgdb.cn/item/64d455e61ddac507cce43876.png)
> 
> 答案：
> 
> 1. 这个结构俗称“bottleneck”。
> 
> 2. 可以强迫网络学习一个更compressed representation。常用于Autoencoder。

> 关于权重初始化的一些定性的建议。
> 
> ![](https://pic.imgdb.cn/item/64d455e61ddac507cce438c5.png)

> 关于如何缓解训练时遇到plateau。
> 
> ![](https://pic.imgdb.cn/item/64d455e71ddac507cce43912.png)