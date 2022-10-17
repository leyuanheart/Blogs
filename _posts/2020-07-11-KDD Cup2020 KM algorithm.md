---
layout:     post
title:      我做KDD比赛时学习KM算法的心路历程
subtitle:   
date:       2020-07-11
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - KDD Cup
    - KM Algorithm
---

这是一篇很长的文章，从最开始了解KM算法的原理，然后在scipy中找到用来计算的函数，再到用cvxpy凸优化库来写KM，之后又因为不能在比赛平台里加载cvxpy库，于是得自己把KM的图论逻辑给写出来，这一段最为反复，前后耗时大概一个星期，我得从头去了解了概念，什么二分图、最大匹配、增广路径、深度优先搜索、广度优先搜索......，然后还得把这些写成代码，参考了非常多的博客、github上的代码（当然最主要的还是搞到了一个比较好的bsfkm），把整个代码逻辑在纸上推演了一遍，总算是成功地写两版KM算法（深度优先和广度优先）。虽然说是写出来了，我觉得我还是没有完全理解，不仅是原理，而且还有代码的设计，所以就打算写一篇文章总结一下我所学到了，希望能加深理解，之后也方便回顾，这也算是我第一次写总结性的文章，之前很多时候都是边学变做笔记，然后学完就觉得差不多懂了，但是我现在回想，很多知识我已经完全没有印象了，说明当时并没有学透，想再去看看，手边又没有资料，全然提不去干劲，于是就这么过去了。所以从这篇文章开始，我会试着培养总结的习惯，不只是记笔记，而是记录下在学习过程中的碰到的困难，自己的想法、心态，最主要的目的是为了自己日后回顾，当然如果能分享一些知识，我觉得就更有意义了。

好了，罗里吧嗦了一大堆，现在开始进入正题，我会按照我学习的进程来介绍，所以有些知识点并不是连贯的（我承认一部分原因是我从头梳理知识点耗费脑子），我估计写完这篇文章也得花很长时间，因为我又忘得差不多了......

## 一、Kuhn-Munkres (KM) algorithm

因为比赛的需要，所以我去了解这个KM算法，首先参考的是这几篇文章：

- [利用python解决指派问题（匈牙利算法）](https://blog.csdn.net/your_answer/article/details/79160045?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-4&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-4)
- [算法学习笔记(5)：匈牙利算法](https://zhuanlan.zhihu.com/p/96229700)
- [超详细！！！匈牙利算法流程以及Python程序实现！！！通俗易懂](https://blog.csdn.net/tommy0095/article/details/104466364/)
- [Munkres' Assignment Algorithm](http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html)

这里面有提到“匈牙利算法”，后面我会介绍，虽然它们两者很有关系，但其实并不是同一个东西，KM算是匈牙利算法的带权值版本。这里就先把KM算法要解决的问题叫做“指派问题”。

指派问题有很多种理解，有图论的，有矩阵的，上面几篇文章里都有涉及到，因为我对图的知识很少，所以看完之后让我能最快速明白这是在干什么以及怎么干的是矩阵的方法，我就先从矩阵的角度来解释这个问题，接下来我会引用一些上述文章里的图片（自己画图太麻烦了）。

![1.png](https://pic.downk.cc/item/5f0965ad14195aa59430a874.png)

其实指派问题理解起来很简单，就是有n个人和m项工作，每个人只能做一项工作，每项工作也只能有一个人来做，给出不同人做不同工作所耗费的时间（cost matrix）或者是所得到的收益（profit matrix），问如何分配能达到花费最小或者收益最大？n和m可以相同，也可以不同，在操作的时候保证n<=m就行了。

这里以上述方阵作为例子阐述算法的流程（当时我也没怎么明白为什么要这么做，但是跟着过了一遍之后，至少明白怎么做了）。文中有说到，首先进行列规约，即每行减去此行最小元素，每一列减去该列最小元素，规约后每行每列中必有0元素出现。接下来进行试指派，也就是**划最少的线覆盖矩阵中全部的0元素**，如果试指派的独立0元素数等于方阵维度则算法结束，如果不等于则需要对矩阵进行调整，重复试指派和调整步骤直到满足算法结束条件。值得一提的是，用矩阵变换求解也有多种实现，主要不同就在于试指派和调整矩阵这块，但万变不离其宗都是为了用最少的线覆盖矩阵中全部的零元素。![2.png](https://pic.downk.cc/item/5f0965ad14195aa59430a876.png)![3](https://pic.downk.cc/item/5f0965ad14195aa59430a87b.png)![4.png](https://pic.downk.cc/item/5f0965ad14195aa59430a87d.png)![5.png](https://pic.downk.cc/item/5f0965ad14195aa59430a87f.png)

当时作者写的代码我也没细看，我就看到下面说其实在python中可以直接调用函数来实现！就是[scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)于是就很高兴地拿来用了。

![6.png](https://pic.downk.cc/item/5f09663314195aa59430cf6f.png)

我发现这个函数不是必须要求是方阵，而是任意的矩阵都行，至此，我的第一阶段学习告一段落。

## 二、用cvxpy来解决这个0-1规划问题

隔了几天继续来写......

由于某些原因，我们认为用上面的`linear_sum_assignement`会很慢（其实还是太天真了......），所以经由老师指点，我又去学习了cvxpy这个python里用来做凸优化的库，MATLAB里也有，叫cvx。开发者都是Stephen Boyd教授的课题组，Stephen Boyd教授是优化界的杠把子，ADMM算法也是他提出来了（因为用过，所以了解了一些，但是想想好像又不是很记得了，之后再写一篇ADMM的文章吧）。

首先参考内容：

- [cvxpy学习笔记](https://blog.csdn.net/fan_h_l/article/details/81981715?utm_source=blogxgwz7)
- [解决anaconda安装cvxpy失败的方法](https://www.cnblogs.com/pengweiblog/p/11710720.html)
- [cvxpy官方介绍](https://www.cvxpy.org/index.html)

如果要学的话，推荐看官方介绍。

好了，回到我们这个问题上来，其实这个指派问题也可以看成是一个0-1的规划问题。

![7.png](https://pic.downk.cc/item/5f09663314195aa59430cf71.png)

哦，对了，我们主要是参考了这篇文章：[Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach](https://dl.acm.org/doi/pdf/10.1145/3219819.3219824)

上图中的$A_{\pi}$就是权重矩阵，要优化的就是$a$，是一个和$A_{\pi}$一样的矩阵，里面元素全为0或者1，然后带有约束，每行和每列的和加起来不超过1，也就是每个行标和列标做多只能有一个匹配，然后我们这个带约束的优化问题用代码表示出来即可，求解就交给cvxpy，假设我们已经有了权重矩阵，先放一个比较好理解的版本：

```python
import scipy.sparse as sp
import cvxpy as cp
import numpy as np

# 假设我们现在有司机和订单的id列表，并且有一些司机和订单匹配估算得到的价格（注意：并不是每个订单和司机都会匹配，有可能有些是没有匹配的，所以这个矩阵是一个稀疏的矩阵，因此要引入sicpy里的sparse来生成稀疏矩阵

row_num = 100  # 司机个数

col_num = 200  # 订单个数

val_num = 1000 # 权重矩阵里有值的个数，也就是初始提供的司机订单配对数

values = 1 + np.random.randn(val_num) * 2  # 随机生成值

rows = np.random.randint(0, row_num, val_num)  # 随机生成初始配对值所对应的横坐标

cols = np.random.randint(0, col_num, val_num)  # 随机生成初始配对值所对应的纵坐标

A_sp = sp.coo_matrix((values, (rows, cols)), shape=(row_num, col_num))  # 生成稀疏矩阵

# cvxpy求解凸优化问题
X = cp.Variable((row_num, col_num), boolean=True)  # 生成一个只有0和1的变量，维数和A_sp相同

obj = cp.Maximize(cp.sum(cp.multiply(A_sp, X)))  # 要优化的目标

constraints = [X @ np.ones((col_num, 1)) <=1, 
               np.ones((1, row_num)) @ X <=1]   # 满足的约束，右边的1其实起到的是向量的作用

problem = cp.Problem(obj, constraints)     # 生成problem

problem.solve()  # 求解

problem.value    # 输出最大值

probelm.X        # 输出对应的X的值
```

这种写法很直接，但是当维数很高时，其实是很慢的，因为变量$X$是一个矩阵，于是就有了第二种写法（并不是我写的，而是我看了别人的代码，只能是感叹，妙啊！），把变量设置成向量，而把约束设置成矩阵：

```python
import scipy.sparse as sp
import cvxpy as cp
import numpy as np

# 假设我们现在有司机和订单的id列表，并且有一些司机和订单匹配估算得到的价格（注意：并不是每个订单和司机都会匹配，有可能有些是没有匹配的，所以这个矩阵是一个稀疏的矩阵，因此要引入sicpy里的sparse来生成稀疏矩阵

row_num = 100  # 司机个数

col_num = 200  # 订单个数

val_num = 1000 # 权重矩阵里有值的个数，也就是初始提供的司机订单配对数

values = 1 + np.random.randn(val_num) * 2  # 随机生成值

rows = np.random.randint(0, row_num, val_num)  # 随机生成初始配对值所对应的横坐标
cols = np.random.randint(0, col_num, val_num)  # 随机生成初始配对值所对应的纵坐标


# cvxpy求解凸优化问题

order_mat_row, order_mat_col = [None] * val_num, [None] * val_num
driver_mat_row, driver_mat_col = [None] * val_num, [None] * val_num

for idx, (did, oid) in enumerate(zip(rows, cols)):
    order_mat_row[idx], order_mat_col[idx] = oid, idx
    driver_mat_row[idx], driver_mat_col[idx] = did, idx

driver_mat = sp.coo_matrix((np.ones(val_num), (driver_mat_row, driver_mat_col)), shape=(row_num, val_num))

order_mat = sp.coo_matrix((np.ones(val_num), (order_mat_row, order_mat_col)), shape=(col_num, val_num))

# 上面这一段相当于建立分别建立司机和订单的稀疏矩阵，以司机为例，这个矩阵行是司机的id，列是所有司机订单的配对值，对于每一行来说，记录的是该司机在哪些配对值中出现过，出现为1，否则为空；订单矩阵的含义类似

# cvxpy求解凸优化问题

X = cp.Variable(val_num, boolean=True)  # 由原来的选哪些司机和订单变成直接是选哪些值

obj = values@X
constr = [
    order_mat @ X <= 1,    # 意味着每个订单只能匹配不超过一次
    
    driver_mat @ X <= 1,   # 每个司机只能匹配不超过一次
]
prob = cp.Problem(cp.Maximize(obj), constr)
prob.solve(solver=cp.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF', 'presolve': 'GLP_ON'})  # 可以暂时忽略这些参数

prob.value
prob.X
```

这么写，算起来就真的是快了，而且甚至超过了`linear_sum_assignment` （感动到不行啊！）

可是，提交的时候发现报错了，找不到模块cvxpy......平台上并不能安装新的库，我们尝试了一下，把本地整个cvxpy的文件夹和代码一起打包，依然不行（我也不知道是我们的将库导入项目的方法不对，还是它那个平台就是不能这么干），所以这条路没走通......于是就进入了下一个阶段，也是让我最想放弃的阶段。

## 三、从图论的逻辑来写KM算法

上回说到从优化的角度虽然可以解决，但是比赛里用不上，所以还是得回到KM算法本身的逻辑，这个时候还是保有着`linear_sum_assignment` 会很慢的思想，又经过老师的指点（这次是直接找了一个图论逻辑写的KM算法的代码，其实上次也是类似了......），然后就开始看代码，理解逻辑，发现完全看不懂啊，这些的什么啊......于是又把一些从图论逻辑介绍KM算法的文章找出来看，结果就是越看发现要补的内容越多，搜的资料也越来越多，看的过程中真的好几次都想着算了，反正有一个现成的代码（虽然不懂，但是能用......），但有时候就是不服气呗，看不懂也忍着看了，看了一天多之后感觉脑子里的内容（虽然我很想称其为知识，但是我觉得不配）慢慢能连得上了，唉，来劲了，虽然后面的过程也不是很轻松顺利，但是每天有一点进步，就有动力继续做，还是和之前一样，先放参考的内容（这次可能有些多，所以我就分部分贴出来......）

- [算法学习笔记(5)：匈牙利算法](https://zhuanlan.zhihu.com/p/96229700)（下面的几幅图都是出自这篇博客）
- [【原创】我的KM算法详解](https://www.cnblogs.com/zpfbuaa/p/7218607.html)
- [匈牙利算法（二分图）](https://www.cnblogs.com/shenben/p/5573788.html)（推荐认真看看这个）

在第一部分说过，匈牙利算法和KM算法是两种不同的算法，其实匈牙利算法主要用于解决一些与**二分图匹配**有关的问题，而KM算法提出是为了解决上面提到的指派问题，而且我估计当时他们就是从矩阵的角度出发的，只不过后来发现其实也可以从带有权值边的二分图来看，将KM理解成了带权值的匈牙利算法。KM算法提出的时间更早，1957年由Kuhn和Munkras提出，而匈牙利算法是在1965年提出的。因此，要想理解KM的图论逻辑，就要先介绍匈牙利算法，再进一步，从**二分图**开始。

### 1、二分图

<img src="https://pic.downk.cc/item/5f09663314195aa59430cf73.png" alt="8.png" style="zoom:50%;" />

**二分图**（**Bipartite graph**）是一类特殊的**图**，它可以被划分为两个部分，每个部分内的点互不相连，两部分之间随意连（有更严格的定义，还有方法判断是不是二分图，我不懂，就不介绍了）。匈牙利算法主要用来解决两个问题：求二分图的**最大匹配数**和**最小点覆盖数**，其实这两个问题是等价的（有**König定理**告诉我们的）。那怎么做呢，主要就是找**增广路径**（又要介绍概念了）。

上图是一个初始连接情况，还没有开始匹配（初始都没连接的肯定是不能匹配的），匹配的含义我再明确一下，就是一对一的关系，我们要做的就是找上图的最大匹配数。假设现在先匹配了一对，我来介绍一下增广路径。

<img src="https://pic.downk.cc/item/5f09663314195aa59430cf76.jpg" alt="9.png" style="zoom:50%;" />

**交替路：**从一个未匹配点出发，依次经过非匹配边、匹配边、非匹配边...形成的路径叫交替路。以上图为例，B2---->G2---->B1就是一条交替路，B2---->G2---->B1---->G4也是一条交替路。

**增广路**：从一个未匹配点出发，走交替路，如果途径另一个未匹配点（出发的点不算），则这条交替路称为增广路（augmenting path）。上图中B2---->G2---->B1---->G4就是一条增广路。

为什么要找增广路呢，我们要注意到增广路的一些特点：

1. 有奇数条边 
2. 起点终点分属二分图的两边
3. 左右两边的点交错出现
4. 整条路径上没有重复的点
5. 起点和终点都是目前还没有配对的点，其他的点都已经出现在匹配子图（这个概念应该好理解）中
6. 路径上的所有第奇数条边都是目前还没有进入目前的匹配子图的边，而所有第偶数条边都已经进入目前的匹配子图

厉害的来了，因为奇数边总是比偶数边多一条，所以，当我们把奇数边的点加入匹配子图，而把偶数边的点删去时（奇偶互换），匹配数就+1了！！！！

于是，匈牙利算法的流程就是，从左边的部分开始，先给一个初始试匹配，然后去寻找是否有增广路径，有的话，奇偶互换，一直到找不到增广路径为止，就得到了最大匹配。

以上图为例来叙述一下从B2出发找增广路径是一种怎样的体验：B2出发，走到G2，发现G2已经和B1匹配了，于是走到B1那，问问B1有没有其他的备选，结果发现B1还有一个b备选G4，于是就就走到G4那，让G4和B1连上，自己就和B2连上了。

下图就是一种最大匹配的结果，可以试试，找不到增广路径了。

<img src="https://pic.downk.cc/item/5f09663314195aa59430cf79.png" alt="10.png" style="zoom:50%;" />

为什么说是一种最大匹配的结果呢，首先因为很明显，不止这一种结果能能到达最大匹配，比如B1---G2，B3---G3，B4---G4就是另一种。出现这样的原因首先和你初始选的匹配有关，其次，在搜索增广路径时是有两种方法的：深度搜索优先（Depth-first search）和广度搜索优先（Breadth-first search）。

啥意思呢，本来我想通过上面的图来直观解释一下，之后再详细介绍，但是我发现由于初始试匹配的选取，导致上面的图显示不出深度和广度优先的差别......，逼得我自己动手改一下这个图<img src="https://pic.downk.cc/item/5f0966d814195aa59431013c.jpg" alt="11.png" style="zoom:33%;" />，现在假设初始的匹配是B2---G2，然后我们从B1出发找增广路径，

1. 深度优先：B1走到G2，发现G2已经和B2匹配了，于是走到B2问有没有其他备选，发现没有其他备选，没有办法，于是自己改道，B1走到G4，发现G4没有被匹配，果断匹配上
2. 广度优先：B1走到G2，发现G2已经和B2匹配了，他不是先找B2，而是先看看自己有没有备选，发现有，于是直接走到G4匹配

深搜就是先一条路走到底，不行了再回来重新找路，而广搜则是一旦这条路卡了，我就先找其他的路。这样解释一遍应该对这两个概念有了大致的理解，我当时也是这样想的，后来我发现理解了是一回事，能不能把它写成代码又是另一回事，深度优先还好，可以由递归过程来实现，而广搜则要自己维护一个队列，并对一路过来的路线自己做标记，当时真是要了我的老命了，虽然最后我写出来，我目前也并不能熟练地运用。

### 2、深度搜索优先（DFS）和广度搜索优先（BFS）

关于DFS和BFS的介绍，包括如何用python来实现，我是参考了这几篇文章

- [python中的 DFS 与 BFS](https://www.jianshu.com/p/b215152a85fb)
- [Understanding the Breadth-First Search with Python](https://medium.com/@yasufumy/algorithm-breadth-first-search-408297a075c9)
- [How to Implement Breadth-First Search in Python](https://pythoninwonderland.wordpress.com/2017/03/18/how-to-implement-breadth-first-search-in-python/)

下面我将从代码实现的角度介绍这两种搜索，介绍、代码和图片都是出自上面3篇文章，我自己的实际体验是，就算这一部分弄懂了，仍然是不足以写出KM算法的......

首先有这么一个图（有顶点和边构成的）

![12.png](https://pic.downk.cc/item/5f0966d814195aa59431013e.png)

把它用python里的字典表示出来，键表示顶点，值表示与键相连的顶点。

```python
graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}
```

我们要用Depth-First和Breath-First搜索来实现下面的目标：

1. 查找所有的连接到目标顶点的点（直接连和间接连都算，这个问题里所有点都是互连的）
2. 返回两个顶点之间的所有可行的路径
3. 在BFS的情况下，返回最短路径（长度为路径边的数目）

下面还会涉及**堆栈**和**队列**的概念，我也不是特别懂，但是形象地解释是OK的，这里的名词使用其实有些混乱，**堆**和**栈**是两个概念，而**栈**又常被称为**堆栈**......

`堆`：就是指在程序运行时，向系统申请某个大小的内存空间，用来按照一定顺序存放数据

`栈`：可以把它理解成一个桶，数据从上面依次扔下来，也只能从上面取，也就是“后进先出”

`队列`：可以把它比喻成左右两端开口的圆柱体，从左边放数据，从右边取数据，也就是“先进先出”

#### 深度优先搜索

回顾一下DFS，在回溯之前探索每个分支的可能顶点。下面的实现使用堆栈数据结构来构建，使用Python里的列表的pop方法从中删除元素，注意我们只能添加未访问的相邻顶点。

```python
def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()  # 默认从最后删除
        
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)   # 这里一定注意要除掉已经访问的顶点，不然会进入死循环
    return visited

dfs(graph, 'A') # {'E', 'D', 'F', 'A', 'C', 'B'}
```

下面这个与上面的实现相同的功能，但是，这次我们使用更简洁的递归形式，这样就不用去构造一个堆栈了。

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

dfs(graph, 'C') # {'E', 'D', 'F', 'A', 'C', 'B'}
```

当时看到这就觉得，递归真是妙啊！以后碰到堆栈的问题就要想想能不能用递归。

我们能够调整前面的两个实现，以返回开始和目标顶点之间的所有可能路径。 下面的实现再次使用堆栈数据结构来迭代地解决问题，在我们找到目标时产生每个可能的路径。使用`生成器`将允许用户得到可能的多条路径。

```python
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]   # 返回一个生成器
                
            else:
                stack.append((next, path+[next]))
                
list(dfs_paths(graph, 'A', 'F'))    # [['A', 'C', 'F'], ['A', 'B', 'E', 'F']]
```

下面的实现使用递归方法调`yield from` [PEP380](http://legacy.python.org/dev/peps/pep-0380/) addition 来返回调用的定位路径。但是原文会报错，说的是当时他的服务器上安装的[Pygments](http://pygments.org/)版本不包含更新的关键字组合，不过我并没有这个问题，应该是版本够新了。（多说一句，其实这个`yield from `远没有看上去的那么简单）

```python
def dfs_paths(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        yield path
    for next in graph[start] - set(path):
        yield from dfs_paths(graph, next, goal, path+[next])
        
list(dfs_paths(graph, 'C', 'F')) # [['C', 'F'], ['C', 'A', 'B', 'E', 'F']]
```

#### 广度优先搜索

终于写到这了（从二分图末尾一段写到这已经花了我2个小时40分钟了......，我去锻炼一会）。BFS增加了保证返回最短路径的能力。使用递归方式来实现BFS有些棘手，所以这里只展示用队列数据结构来实现（原文翻译是这么说的，而且我也不会......)。与迭代DFS实现类似，唯一需要的更改是从列表结构的开头而不是最后一个堆栈中删除下一个项目。

```python
def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)  # 表示删除第一个
        
        if vertex not in visited:
               visited.add(vertex)
               queue.extend(graph[vertex]-visited)
    return visited

bfs(graph, 'A') # {'B', 'C', 'A', 'F', 'D', 'E'}
```

有一个小技巧：为了使代码更有效率， 可以使用 `collections`里面的`deque`对象，然后用popleft( )方法去替代pop(0)，原因是popleft( )的时间复杂度是O(1)，而pop(0)是O(n)。（虽然我也不知道为什么）

```python
from collections import deque

def bfs(graph, start):
    visited= set()
    queue = deque([start])
    while queue:
        vertex = queue.popleft()  # 表示删除第一个
        
        if vertex not in visited:
               visited.add(vertex)
               queue.extend(graph[vertex]-visited)
    return visited

bfs(graph, 'A') # {'B', 'C', 'A', 'F', 'D', 'E'}
```

可以再次稍微改变该实现，以返回两个顶点之间的所有可能路径，其中第一个是最短的这种路径之一。（这里就不能用deque类了）

```python
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))
                             
list(bfs_paths(graph, 'A', 'F'))   # [['A', 'C', 'F'], ['A', 'B', 'E', 'F']] 
```

知道BFS路径生成器方法首先会返回最短路径，我们可以创建一个有用的方法，它只返回找到的最短路径，如果没有路径，则返回“None”。

```python
def shortest_path(graph, start, goal):
    try:
        return next(bfs_paths(graph, start, goal))
    except StopIteration:
        return None
    
shortest_path(graph, 'A', 'F') # ['A', 'C', 'F']
```

至此，两种搜索方法就讲完了，终于要进入最终篇章了......

### 3、python实现DFSKM和BFSKM

距离上次写又有一段时间了，我还得把前面的内容重看一遍，再往下继续写（还是没有懂透啊）。

先放参考的文章：

- [匈牙利算法(Kuhn-Munkres)算法](https://blog.csdn.net/zsfcg/article/details/20738027)
- [我对KM算法的理解](https://www.iteye.com/blog/philoscience-1754498)
- [【原创】我的KM算法详解](https://www.cnblogs.com/zpfbuaa/p/7218607.html)
- [python实现KM算法](https://blog.csdn.net/lcl497049972/article/details/91138040)
- [【原创】KM算法的Bfs写法](https://blog.csdn.net/c20182030/article/details/73330556?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1)
- [二分图最大权匹配【KM算法 BFS优化下的真正的O(N3)的KM算法】【KM算法模板】](https://blog.csdn.net/qq_41730082/article/details/103396480?depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1&utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1)

还记得之前说过的匈牙利算法和KM算法的联系吗，对于一个不带权重的二分图，我们用匈牙利算法去找它的最大匹配，也就是找增广路径的过程。对于带权重的二分图，我们的目标是要找到权值最优的匹配，稍微想一下就明白无法直接用匈牙利算法，于是要怎么做呢？KM算法采用了一种十分巧妙的方法（也是KM算法的精髓），就是引入顶标（标杆），将最优权值匹配转化为最大匹配问题。

什么意思呢，下面我用具体例子来演示一下算法的流程。

![13.png](https://pic.downk.cc/item/5f0966d814195aa594310140.png)

假设我们有这么一个权重矩阵（这个矩阵也可以是稀疏的，就是存在有些位置没有值），我们对每个点$X_i$和右边每个点$Y_i$添加顶标$C_x$和$C_y$。其中我们要满足$C_x+C_y>=w_{xy}$（$w_{xy}$即为点$X_i$和$Y_i$之间的边权值），这是顶标添加要满足的条件。

接下来介绍一个**相等子图**的概念，就是指在给定的图（包含点和边），要满足$C_x+C_y=w_{xy}$，每次寻找增广路径只在相等子图里寻找。

然后我们就要先初始化一个相等子图，怎么做呢，既然是要找最优权值匹配（这里以最大为例），那么我们就找每一行最大的元素对应的点连成边（以X还是以Y作为分析对象是等价的），然后$C_x=max(w_{xy})$，$C_y=0$。

![14.png](https://pic.downk.cc/item/5f0966d814195aa594310143.png)

如果这张初始化的相等子图满足一一对应的关系，则最优权值匹配就已经找到了，但通常不会这么顺利，会出现上图这样几个X对应一个Y，或者有的Y没被连起来的情况，这个时候就要开始算法的流程了。

首先从X0开始，找增广路，X0---Y1，成了，连起来

![15.png](https://pic.downk.cc/item/5f0966d814195aa594310148.png)



然后从X1出发，找增广路，走到Y0，发现Y0已经被X0匹配了，然后就走向X0（这个时候它是有两种选择的，直接走向X0和先看自己有没有其他的可行路径，没有再走向X0，这就是深搜和广搜的区别，如果你没明白这里我在说什么，那么前面可能没有理解透彻），发现X0并没有其他可行路径，于是X2宣告找不到增广路径。

这个时候就要发动顶标的作用了，通过更改顶标引入新的边（如果一直能找到增广路径就不需要改变顶标），怎么改？

我们先有一个直观的印象，就是如果就按照现在这个情况分配，那么得到的匹配权值结果是6+5+4=15，但是这个最大的权值是无法实现的，所以我们得退而求其次，找比它小的权值，为了能够在相等子图中引入边，考虑到最开始我们取的是每行的最大，举个例子，比如要想把X0---Y1这条边引进来，现在顶标和$6+0>w_{01}=2$，所以所有要么X0的顶标减小，要么Y1的顶标增大，或者同时，才有可能把这条边引进来对吧。

每次更改的原则是：小是肯定要减小的，但是我找减小的最少的那条边加进来，怎么做，先放图，我再解释

![16.png](https://pic.downk.cc/item/5f09675b14195aa594312b7d.png)

S和T分别表示已经搜索过的X和Y的集合（也可以理解为是在交替路径上的集合），我们计算一个值d（图里是$\alpha_l$），就是所有在S里但是不在T的X、Y的顶标和与对应的权重之差的最小值，然后所有在S里的X减去d，所有在T里的Y加上d，其他的顶标保持不变。

![17.png](https://pic.downk.cc/item/5f09675b14195aa594312b7f.png)

你会发现通过这样操作之后，原来在相等子图里的仍然在相等子图里（一加一减嘛），然后新的边X0---Y2和X1---Y2也被引入相等子图里了，再为X1重新寻找增广路径，这里给出两种搜索方法的结果（左：深搜，右：广搜）。

![18.png](https://pic.downk.cc/item/5f09675b14195aa594312b82.png)

你会发现两种方法得到的X0和X1的匹配权值和都是8，比最开始X0和X1都匹配Y0得到的11正好少了d=3。

之后操作就是一样的，从X3出发，找增广路，如果找到，则算法结束，得到的就是最优权值匹配，如果找不到，就更改顶标再找，如果还找不到，就再改，再找，一定是可以找到的（前提是X的个数是小于等于Y的个数）。

理解到这，就可以开始写了，想着两种搜索方法，都说深搜好写，那就先写深搜，确实写得还算顺利，参照各种代码，然后写出来的，也能正确运行，但是当维数很大的时候，发现巨慢......，于是准备写广搜的方法，本以为会很快写完，然而还是too young，too simple。好了我又要去运动了。

我回来了，接下来开始写增广路径用深度优先搜索的KM算法，先把上面例子深搜的完整推导给出来：

![19.png](https://pic.downk.cc/item/5f09675b14195aa594312b84.png)

#### DFSKM代码

```python
# 假设我们有一个n*m的profit矩阵，n<=m，实际操作的时候，永远把较小的作为行就可以了，当然你也可以在函数里通过判断行列大小来写一个变换的操作，这样就不用考虑矩阵的形状了，但是这里就不再增加工作量了，默认在函数外我就已经变换好了

import numpy as np

def km_dfs(profit): 
    
    nrows, ncols = profit.shape
    # 初始化顶标
    
    left_top_idx = np.max(profit, axis=1).tolist()   # 左边就是上面例子中的上方，取每行最大值
    
    right_top_idx = [0] * ncols                      # 右边就是上面例子中的下方，赋值0
    
    left_match = [None] * nrows        # 用来记录每个行标对应的匹配列标
    
    right_match = [None] * ncols       # 用来记录每个列标对应的匹配行标，如果列大于行，会留有None
    
    
    def find_aug_path_dfs(i):
        '''这里寻找增广路径是对于每个行标而言的，因为我们是要遍历行标，所以每个行标都要走一遍这个搜索流程，
        
        并且用了递归来维护堆栈数据，回想之前提到过的搜索连接节点的问题，我们只添加未搜索过的节点，不然会进入死循环
        '''
        
        left_visit[i] = True
        for j, weight in enumerate(profit[i]):
            if right_visit[j]: continue # 已被访问（解决递归中的冲突）
                
            # RecursionError: maximum recursion depth exceeded while calling a Python object
            
            gap = left_top_idx[i] + right_top_idx[j] - profit[i][j]  # 计算gap是用来判断哪个Y是在相等子图里的，并且记录非0的gap值用于更新顶标
            
            if gap == 0:  # abs(gap) < zero_threshold
                
                right_visit[j] = True
                
                if right_match[j] == None or find_aug_path_dfs(right_match[j]): 
                # j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选项
                
                    left_match[i] = j                    
                    right_match[j] = i

                    return True
            else:
                
                if gap < slack[j]: slack[j] = gap   # 通过这一步将递归中的slack值不断更新，保持对于每个列标来说我都是取了最小的那个（虽然可能不是同一个i对应的j）

        return False
    

    for i in range(nrows):   # 遍历每一个行标
        
        while True:
            slack = [np.inf] * ncols    # 初始化slack用于记录gap值
            
            # 初始化搜索指标，每搜索一个行标都要重新初始化visited集合，在更新完顶标之后，即使是同一个行标，也是要重新初始化visited集合的，这个逻辑要理清
            
            left_visit = [False] * nrows
            right_visit = [False] * ncols
            
            if find_aug_path_dfs(i): break  # 如果找到增广路径，则跳出while，进行下一个行标的搜索，但是如果没找到增广路径，即返回的是False，则进行下面的更新顶标操作，并且依然在while循环里，为该行标寻找增广路径，如果依然是False，再修改，直到返回True
                
            d = np.inf
            for j, slk in enumerate(slack):
              if not right_visit[j] and slk < d:
                d = slk         # 考虑没有被搜索过的Y当中最小的gap，因为只要是调用find_aug_path_dfs()函数，那么该行标X就会被标记为访问过
                
            for r in range(nrows):
                if left_visit[r]: left_top_idx[r] -= d
            for c in range(ncols):
                if right_visit[c]: right_top_idx[c] += d
       
    value = 0
    for i, j in enumerate(left_match):
        value += profit[i][j]
        
    return left_match, value


profit = np.array([[3, 4, 6, 4, 9], 
                   [6, 4, 5, 3, 8],
                   [7, 5, 3, 4, 2],
                   [6, 3, 2, 2, 5],
                   [8, 4, 5, 4, 7]])

km_dfs(profit)
# ([4, 2, 3, 1, 0], 29)
```

之前提到了，当矩阵维度很大时，这个算法就很慢，$100\times 100$的矩阵就要算好久，我都没等它算完就停止了，而用`scipy.optimize.linear_sum_assignment`和`cvxpy`就是一瞬间的事，这可不行......

于是就有了写广搜KM的过程，这里要提一下就是我找的很多资料，提到广搜KM的基本都是用C++来写的（我就没找到用其他语言的），深搜KM倒是有一些python代码实现，那我写出来岂不是说明我还可以了，其实也不是，这又要归功于老师了，从他那里得到了一版用BFS写的KM（就是之前说过的用图论逻辑写的KM），然后我利用上面提到的例子按照它的逻辑在纸上一步步过代码，过完才弄清楚代码的逻辑，然后仿照着自己写了一版出来，虽然我看懂了逻辑，但是要让我自己构思出这样一套逻辑，我目前还做不到，可见自己还弱得很，好了回归正题，这次是真真正正进入最终章了。

先把上面例子广搜的完整推导给出来：

![20.png](https://pic.downk.cc/item/5f09675b14195aa594312b87.png)

#### BFSKM代码

先放一个我自己写的第一版，我认为是按照广搜的逻辑来写了，但写出来和深搜一样慢。。。

```python
def km_bfs(profit):
    nrows, ncols = profit.shape
    # 初始化顶标
    
    left_top_idx = np.max(profit, axis=1).tolist()
    right_top_idx = [0] * ncols
    
    left_match = [None] * nrows
    right_match = [None] * ncols

    def find_aug_path_bfs(i):
        left_visit[i] = True
        queue = []  # 记录那些对于第i个节点来说所有可能的匹配节点（即所有有边的节点）
        
        for j, weight in enumerate(profit[i]):
            gap = left_top_idx[i] + right_top_idx[j] - profit[i][j]
            if right_visit[j]: continue  # 防止无穷递归
                
            if gap == 0:
                right_visit[j] = True
                queue.append(j)
                if right_match[j] == None:   # 如果j节点没有被匹配过，那么配对成功，否则就要往下走，继续搜索对于i来说其他可能的节点
                    
                    right_match[j] = i
                    left_match[i] = j
#                    print(i, j, 'success')

                    return True          
            else:
                slack.append(gap)
        for j in queue:                  # 这里就是搜索对i来说可能的节点，如果能搜到，则完成匹配，不能的话就说明找不到增广路径，需要修改顶标
            
            if find_aug_path_bfs(right_match[j]):
                right_match[j] = i
                left_match[i] = j
#                print(i, j, 'success')

                return True
        
#        print(i, 'fail')

        return False

    for i in range(nrows):
        while True:
            # 初始化搜索指标
            
            left_visit = [False] * nrows
            right_visit = [False] * ncols 
            slack = []
        
            if find_aug_path_bfs(i): break
            d = min(slack)
            for r in range(nrows):
                if left_visit[r]: left_top_idx[r] -= d
            for c in range(ncols):
                if right_visit[c]: right_top_idx[c] += d

    value = 0
    for i, j in enumerate(left_match):
        value += profit[i][j]
    
    return left_match, value


profit = np.array([[3, 4, 6, 4, 9], 
                   [6, 4, 5, 3, 8],
                   [7, 5, 3, 4, 2],
                   [6, 3, 2, 2, 5],
                   [8, 4, 5, 4, 7]])

km_bfs(profit)
# ([4, 2, 1, 0, 3], 29)
```

我觉得我理解得没问题，结果也是正确的，但是不知道是不是使用递归的原因，导致这一版的算法仍然很慢。接下来，就是高能时刻了，我理解的它变快的原因是，它里面也用了一个队列queue，但是这个队列存的是当次没有被匹配上的y上一次对应的x，还用了一个pre，用来存储slack是对应于哪一个x的匹配，这样在之后需要重新给之前的节点分配指标的时候就可以直接索引，而不用再进行一次搜索增广路径，从而大大加快了速度，说起来没有什么感觉，还是来看看代码吧。

```python
# from collections import deque   # 使用deque的话就把相应的注释给去掉，同时把上一行给注释掉

def km_bfs(profit):
    
    nrows, ncols = profit.shape
    # 初始化顶标
    
    left_top_idx = np.max(profit, axis=1).tolist()
    right_top_idx = [0] * ncols
    
    left_match = [None] * nrows
    right_match = [None] * ncols
    
    
    
    def find_aug_path_bfs(s):
        
        slack = [np.inf] * ncols
        pre = [None] * ncols   # 用来记录slack是对应于哪个一个x的匹配
        
        # 初始化搜索指标
        
        left_visit = [None] * nrows
        right_visit = [None] * ncols
        
        queue = [s] # 建立队列，用来记录当次没有被匹配上的y上一次对应的x，将要搜寻的x作为初始值
        
#        queue = deque([s]) 

        left_visit[s] = True
        
        
        def check(j):
            right_visit[j] = True
            
            if right_match[j] != None:
                queue.append(right_match[j])
                left_visit[right_match[j]] = True
                return False
            
            while j != None:
                right_match[j] = pre[j]
                tmp = left_match[pre[j]]
                left_match[pre[j]] = j
                j = tmp
            return True
            
            
        while True:
            while queue:
                i = queue.pop(0)     # i = queue.popleft()
                
                for j in range(ncols):
                    gap = left_top_idx[i] + right_top_idx[j] - profit[i][j]
                    if not right_visit[j] and slack[j] >= gap:
                        pre[j] = i
                        if gap != 0:
                            slack[j] = gap
                        elif check(j):
                            return
        
            delta = np.inf
            for j in range(ncols):
                if not right_visit[j] and delta >= slack[j]:
                    delta = slack[j]
            
            for j in range(ncols):
                if right_visit[j]:
                    right_top_idx[j] += delta
                else:
                    slack[j] -= delta
            
            for i in range(nrows):
                if left_visit[i]:
                    left_top_idx[i] -= delta
            
            for j in range(ncols):
                if not right_visit[j] and slack[j] == 0 and check(j):
                    return
          
    for i in range(nrows):
        find_aug_path_bfs(i)
    
    value = 0
    for i, j in enumerate(left_match):
        value += profit[i][j]
        
    return left_match, value


profit = np.array([[3, 4, 6, 4, 9], 
                   [6, 4, 5, 3, 8],
                   [7, 5, 3, 4, 2],
                   [6, 3, 2, 2, 5],
                   [8, 4, 5, 4, 7]])

km_bfs(profit)
# ([2, 4, 3, 1, 0], 29)
```

老实说现在写出来我还是不怎么能看明白这个算法的逻辑，所以虽然会花很长时间，但是还是再把这个算法在这里手敲一遍流程，就用这一节的例子：

![21.png](https://pic.downk.cc/item/5f0967b614195aa5943146a2.png)

![22.png](https://pic.downk.cc/item/5f0967b614195aa5943146a4.png)

![23.png](https://pic.downk.cc/item/5f0967b614195aa5943146a8.png)

![24.png](https://pic.downk.cc/item/5f0967b614195aa5943146ab.png)

![25.png](https://pic.downk.cc/item/5f0967b614195aa5943146b0.png)

![26.png](https://pic.downk.cc/item/5f09680014195aa594315d9d.png)

![27.png](https://pic.downk.cc/item/5f09680014195aa594315d9f.png)

![28.png](https://pic.downk.cc/item/5f09680014195aa594315da1.png)

![29.png](https://pic.downk.cc/item/5f09680014195aa594315da3.png)

![30.png](https://pic.downk.cc/item/5f09680014195aa594315da8.png)

![31.png](https://pic.downk.cc/item/5f09683714195aa594317413.png)

![32.png](https://pic.downk.cc/item/5f09683714195aa594317415.png)

![33.png](https://pic.downk.cc/item/5f09683714195aa594317417.png)

这段BFS代码我写了将近4个小时了......，

但是至此这篇文章终于写完了，从最开始写到完成已经快两个月过去了......，中间很多事打断，时不时想起来还没写完就又开始写，一次30分钟到2个多小时不等，累积时长我记录了一下，13小时30分钟。



------

后话，这段写于比赛结束之后几天，简单介绍一下我们所做的工作，我们使用的策略是参考上面那篇调度的文章，使用强化学习在reward中加入value function（考虑长期受益）作为KM算法的权重，一开始我们是以Value table的形式，使用TD update来估计value 函数，但是算出来的结果加入原来的算法中表现还没有不加value函数好，后来我们又尝试用Neural Network去拟合，但是这一部分不是我写的，而且到后面写这一部分的同学入职了。。。，所以到最后一个星期其实我们已经处于放弃状态了，我当时看了一下我们的排名，派单的话在79名，调度的话没看，但应该也很后（一共是大概1000只队伍，有单人也有团队的），然后今天再看的时候，发现我们最终的排名，派单41名，调度38名，还往上涨了。。。，应该是有些队伍的算法过拟合了，在最后的评估阶段分数掉下来了（后来我又想到很可能是很多排在我们前面的个人参赛者合并成一支队伍，嗯，这应该是最主要的原因）。

嗯，这是一个反面例子，做事还是要有始有终。

