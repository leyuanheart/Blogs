---
layout:     post
title:      Deep learning from scratch 2
subtitle:   Neural Network Learning
date:       2021-02-14
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Deep Learning
---

<img src="https://img.imgdb.cn/item/60311e2d5f4313ce25b859d6.jpg" alt="fengmian" style="zoom:50%;" />

## 神经网络的学习

这里所说的“学习”是指从训练数据中自动获取最优权重参数的过程。为了使神经网络能进行学习，将引入**损失函数**这一指标。

### 从数据中学习

![21](https://img.imgdb.cn/item/6031c16b5f4313ce25e9a721.png)

深度学习有时也成为了端到端机器学习（end-to-end machine learning），也就是从原始数据中获得目标结果的过程，中间不需要人为的介入。

#### 训练数据和测试数据

机器学习中，一般将数据分为**训练数据**和**测试数据**两部分来进行学习和实验。首先，使用训练数据进行学习，寻找最优参数；然后，使用测试数据评价得到的模型的实际能力。为什么要划分呢？因为我们追求的是模型的**泛化能力**。

泛化能力是指处理未被观察过的数据（不包含在训练数据中的数据）的能力。获得泛化能力是机器学习的最终目标。因此，仅仅用一个数据集去学习和评价参数，是无法进行正确的评价的。这样会导致可以顺利处理某个数据集，但是无法处理其他数据集的情况。这种只对某个数据集过度拟合的状态称为**过拟合**（over fitting）。

#### 损失函数

神经网络的学习通过某个指标表示现在的状态，然后，以这个指标为基准，寻找最优权重参数。这个指标就被称为**损失函数**（loss function）。这个损失函数可以使用任意函数，但一般用均方误差和交叉熵等。

##### 均方误差（mean square error）

可以用作损失函数的函数有很多，其中最有名的是**均方误差**（mean square error）：


$$
E=\frac{1}{2}\sum_{k=1}^K(y_k-t_k)^2
$$


这里$y_k$表示神经网络的输出，$t_k$表示训练数据的标签（one-hot表示），$k$表示数据的维数。比如在手写数字（0到9）的例子中， $y_k$、$t_k$是由如下10个元素构成的数据。

`y=[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]`

`t=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`，这就是**one-hot表示**。

```python
def mean_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)
```



##### 交叉熵误差（cross entropy error）

除了均方误差外，**交叉熵误差**（cross entropy error）也经常被用作损失函数：


$$
E=-\sum_{k=1}^Kt_k\log y_k
$$


由于$t_k$是one-hot表示，只有正确解标签的索引为1，其他均为0。所以，实际上上式只计算对应正确解标签的输出的自然对数。比如，假设正确解标签的索引是“2”，与之对应的神经网络输出是0.6，则交叉熵误差是$-\log 0.6=0.51$，如果“2”对应的输出为0.1，则误差为$\log 0.1=2.3$。也就是说，**交叉熵误差的值是由正确解标签所对应的输出结果决定的**。

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return - np.sum(t * (np.log(y + delta)))
```

函数内部在计算`np.log`时，加上了一个微小值delta。这是因为，当出现`np.log(0)`时，会变成$-\inf$，这样一来后续计算就无法进行，所以作为保护性对策，加一个微小值防止负无穷的发生。

##### mini-batch学习

机器学习使用训练数据进行学习。因此，计算损失函数是必须将所有训练数据作为对象。也就是说，如果训练数据有100个的话，我们就要把这100个损失函数的总和作为学习的指标。

之前的数学表达是针对单个数据的损失函数，如果是考虑整个数据集，以交叉熵为例，则可以写成：
$$
E=-\frac{1}{N}\sum_{n=1}^N\sum_{k=1}^Kt_{nk}\log y_{nk}
$$
这里假设数据有$N$个，并且最后除以了$N$进行正规化，表示单个数据的”平均损失函数”。通过这样的平均化，可以获得和训练数据的数量无关的统一指标。

另外，如果是用MNIST数据集的话，它的训练数据有60000个，若以全部数据为对象求损失函数的和，则计算过程需要花费较长的时间。再者，如果遇到大数据，数量会有几百万、几千万之多，这种情况下以全部数据为对象计算损失函数是不现实的。因此，我们从全部数据中选出一部分，作为全部数据的“近似”。神经网络的学习也是从训练数据中选出一批数据（称为mini-batch，小批量），然后对mini-batch进行学习。比如从60000个中随机选择100个，再用着100个进行学习。这种学习方式称为**mini-batch学习**。

这里实现一版可以同时处理单个数据和批量数据的交叉熵函数。

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
        batch_size = y.shape[0]
        return - np.sum(t * np.log(y + 1e-7)) / batch_size
```

当标签形式不是one-hot表示，而是像“2”、“7”这样的标签时，交叉熵可以通过如下代码实现，只需要获取神经网络在正确解标签处的输出就行了。

```python
def cross_entropy_error(y, t):
	if y.ndim == 1:
	t = t.reshape(1, t.size)
	y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
##### 为何要设定损失函数

可能有人要问：“为什么要引入损失函数呢？”以手写数字识别任务为例，我们想要获得的是能提高识别精度的参数，特意再导入一个损失函数不是有些重复劳动吗？也就是说，既然我们的目标是获得使识别精度尽可能高的神经网络，那不是应该把识别精度作为指标吗？

之所以不能用识别精度作为指标，是因为这样一来绝大多数地方的梯度都会变成0，导致参数无法更新。我们可以看一个具体的例子，假设某个神经网络正确识别了100个样本中的32个，此时识别精度是32%。如果以识别精度为指标，即使稍微改变权重参数的值，识别精度也仍将保持在32%，不会出现变化。也就是说，仅仅微调参数，是无法改善识别精度的。即便识别精度有所改善，它的值也不会像32.0123...%这样连续变化，而是变成33%、34%这样不连续的、离散的值。而如果把损失函数作为指标，则当前损失函数的值可以表示为0.925432...这样的值。并且，如果稍微改变一下参数值，对应的损失函数也会像0.93432...这样发生连续的变化。

出于同样的原因，如果使用阶跃函数作为激活函数，神经网络将无法学习，因为参数的微小变化也会被阶跃函数抹杀，导致损失函数不会产生任何变化。

### 数值微分

这一部分书中是从导数、偏导数、梯度的定义开始介绍的，我这里就是重点关注一下Implementation。对于复杂的函数，无法通过推导数学公式获得解析解（analytic solution），转而需要利用微小的差分来求导数，这个过程就叫做**数值微分**（numerical differentiation）。

我们根据导数的定义来做数值微分：


$$
\frac{df(x)}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

```python
# 不好的实现
def numerical_diff(f, x):
    h = 1e-50
    return (f(x+h) - f(x)) / h
```

在上面的实例中，因为想把尽可能小的值赋给h（可以的话，想让h无限接近0），所以h使用的1e-50这个微小值。单数这样反而产生了**舍入误差**（rounding error）。所谓舍入误差，是指因省略小数的精细部分的数值（比如，小数点后第8位以后的数值）而造成的计算结果上的误差。比如，在Python中，舍入误差可以表示为：

`np.float32(1e-50)`

`0.0`

如上所示，如果用float32类型（32位的浮点数）来表示1e-50，就会变成0.0。这是第一个需要改进的地方，将h改为1e-4就可以得到正确的结果。

第二个需要改进的地方与函数的差分有关。虽然上述实现中计算了f在x+h和x之间的差分（前向差分），但是，这个计算从一开始就有误差。这个差异出现的原因是h不可能无限接近于0。为了减少这个误差，我们可以计算f在x+h和x-h之间的差分（中心差分）。

![22](https://img.imgdb.cn/item/6031c16b5f4313ce25e9a724.png)

```python
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    
    return (f(x+h) - f(x-h)) / (2*h)
```

梯度（gradient）就是由全部变量的偏导数汇总而成的向量。这里$x$是一个p维向量$(x_1, \ldots, x_p)$，函数f是关于$x$的函数$f(x)$。可以这样实现：

```python
def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(x)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
            
        return grad
    
    
    
'''
上面的numerical_gradient()最多只能处理X是矩阵（2维数组）的情况，如果X是多维数组的话，
比如在CNN中卷积层的W一般是(filter_num, img_channel, height, width)，这个时候就要用到Numpy中的数组迭代器np.nditer()，这个可以迭代多维数组，之后就统一用下面的函数了

'''
def numerical_gradient(f, x):
    '''
    NumPy 迭代器对象 numpy.nditer 提供了一种灵活访问一个或者多个数组元素的方式。
    https://blog.csdn.net/m0_37393514/article/details/79563776
    '''
    h = 1e-4 # 0.0001
    
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val 
        it.iternext()   
        
    return grad
```

举一个实际例子：$f(x_0, x_1) = x_0^2+x_1^2$。我们把该函数在各个点的**负**梯度在图上画出来。

```python
import numpy as np
import matplotlib.pylab as plt
def func(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
    
    
x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(func, np.array([X, Y]).T).T

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.draw()
plt.show()
```

![23](https://img.imgdb.cn/item/6031c16b5f4313ce25e9a728.png)

可以看到，上图中箭头的方向就是负梯度的方向，可以看出，这些方向都指向$f(x_0,x_1)$的“最低处”（最小值），其次我们发现离“最低处”越远，箭头越大。虽然上图中的方向指向了最小值，但并非任何时候都是这样。实际上，负梯度会指向**各点处**的函数值降低的方向。更严格地讲，是各点处函数值减小最多的方向（高等数学告诉我们，方向导数=$\cos \theta \times$梯度（$\theta$是方向导数的方向与梯度方向的夹角）， 因此，在所有下降方向中，负梯度方向下降的最多）。

#### 神经网络的梯度

神经网络的学习也要求梯度。这里所说的梯度是指损失函数关于权重参数的梯度。比如，有一个形状为 $2 \times 3$ 的权重 $W$ 的神经网络，损失函数用 $L$ 表示。此时，梯 度可以用 $\frac{\partial L}{\partial W}$ 表示。用数学式表示的话，如下所示。


$$
\begin{array}{c}

\boldsymbol{W}=\left(\begin{array}{ccc}

w_{11} & w_{12} & w_{13} \\

w_{21} & w_{22} & w_{23}

\end{array}\right) \\

\frac{\partial L}{\partial \boldsymbol{W}}=\left(\begin{array}{ccc}

\frac{\partial L}{\partial u_{11}} & \frac{\partial L}{\partial w_{2}} & \frac{\partial L}{\partial w_{L^{3}}} \\

\frac{\partial L}{\partial u_{21}} & \frac{\partial L}{\partial w_{22}} & \frac{\partial L}{\partial w_{23}}

\end{array}\right)

\end{array}
$$


$\frac{\partial L}{\partial W}$ 的元素由各个元素关于 $W$ 的偏导数构成。比如，第 1 行第 1 列的元素 $\frac{\partial L}{\partial w_{11}}$ 表示当 $w_{11}$ 稍微变化时，损失函数 $L$ 会发生多大变化。这里的重点是，$\frac{\partial L}{\partial W}$ 的形状和 $W$ 相同。

下面，我们以一个简单的神经网络为例，来实现求梯度的代码。为此，我们要实现一个名为 SimpleNet 的类。

```python
def softmax(x):
    if x.ndim == 2:  #[batch_size, p]
        
        x = x - np.max(x, axis=1)
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        return y

    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 将one-hot coding转化成单个coding
    
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SimpleNet(object):
    def __init__(self):
        self.W = np.random.randn(2, 3) # 用高斯分布对权重进行初始化
        
    def predict(self, x):    # x是p=2的向量，也可以是一个batch的数据，[batch_size, 2]
        
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy(y, t)
        
        return loss
```

SimpleNet 类只有 个实例变量，即形状为 $2 \times 3$ 的权重参数。它有两个方法，一个是用于预测的 $\operatorname{predict}(x),$ 另一个是用于求损失函数值的 loss $(x, t)$ 。这里参数 $x$ 接收输入数据，t 接收正确解标签。现在我们来试着用一下这个 SimpleNet。

![25](https://img.imgdb.cn/item/6031c16b5f4313ce25e9a731.png)

接下来求梯度。和前面一样，我们使用 numerical_gradient $(f, x)$ 求梯度（这里定义的函数 $f($ W $)$ 的参数 $W$ 是一个伪参数。 因为 numerical_gradient $(f, x)$ 会在内部执行 $f(x)$, 为了与之兼容而定义了 $f(W)$  ) 。

![26](https://img.imgdb.cn/item/6031c1d35f4313ce25e9cf00.png)

numerical_gradient $(f, x)$ 的参数 $f$ 是函数, $x$ 是传给函数 $f$ 的参数。 因此，这 里参数 $x$ 取 net. W，并定义一个计算损失函数的新函数 $f,$ 然后把这个新定义的函数传递给 numerical_gradient $(f, x)$ 。

#### 梯度法

我们所说的最优参数是指损失函数最小时的参数。但是，一般而言，损失函数很复杂，参数空间庞大，我们不知道它在何处能取得最小。梯度法可以来寻找函数最小值（或者尽可能小的值）。以下所说“梯度方向”都是指“负梯度方向”。

这里需要注意的是，梯度表示的是各点处的函数值减小最多的方向。因此，无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向。实际上，在复杂的函数中, 梯度指示的方向基本上都不是函数值最小处。

> 函数的极小值、最小值以及被称为鞍点（saddle point）的地方，梯度为 0 。极小值是局部最小值，也就是限定在某个范围内的最小值。鞍点是从某个方向上 看是极大值，从另一个方向上看则是极小值的点。虽然梯度法是要寻找梯度为 0 的地方，但是那个地方不一定就是最小值（也有可能是极小值或者鞍点）。此外，当函数很复杂且呈扁平状时，学习可能会进入一个（几乎）平坦的地区，陷入被称为“学习 高原”的无法前进的停滞期。

虽然梯度的方向并不一定指向最小值，但沿着它的方向能够最大限度地减小函数的 值。因此，在寻找函数的最小值（或者尽可能小的值）的位置的任务中，要以梯度的信息为线索，决定前进的方向。

此时梯度法就派上用场了。在梯度法中，函数的取值从当前位置沿着梯度方向前进一定距离，然后在新的地方重新求梯度，再沿着新梯度方向前进，如此反复，不断地沿梯度方向前进。像这样，通过不断地沿梯度方向前进，逐渐减小函数值的过程就是**梯度法** (gradient method) 。

现在，我们尝试用数学式来表示梯度法。


$$
\begin{array}{l}

x_{0}=x_{0}-\eta \frac{\partial f}{\partial x_{0}} \\

x_{1}=x_{1}-\eta \frac{\partial f}{\partial x_{1}}

\end{array}
$$



上式的 $\eta$ 表示更新量，在神经网络的学习中，称为**学习率**（learning rate）。学 习率决定在一次学习中，应该学习多少，以及在多大程度上更新参数。

上式是表示更新一次的式子，这个步骤会反复执行。通过反复执行此步骤，逐渐减小函数值。虽然这里只展示了有两个变量时的更新过程，但是即便增加变量的数量，也可以通过类似的式子（各个变量的偏导数）进行更新。

学习率需要事先确定为某个值, 比如 0.01 或 0.001 。一般而言，这个值过大或过小, 都无法抵达一个“好的位置”。在神经网络的学习中，一般会一边改变学习率的值，一边确认学习是否正确进行了。

下面，我们用Python来实现梯度下降法。

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x) # 之前已经定义过了
        
        x -= lr * grad
        
    return x
```

参数 $f$ 是要进行最优化的函数，init_x 是初始值, lr 是学习率 learning rate，step_num 是梯度法的重复次数。numerical_gradient $(f, x)$ 会求函数的梯度，用该梯度乘以学习率得到的值进行更新操作，由 step_num 指定重复的次数。

我们同样利用函数$f(x_0,x_1)=x_0^2+x_1^2$来做例子展示，因为要画图，所以`gradient_descent()`要做一些修改。

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

x0 = np.arange(-4, 4, 0.25)
x1 = np.arange(-4, 4, 0.25)
X0, X1 = np.meshgrid(x0, x1)
Y = function_2(np.array([X0,X1]))

plt.figure(figsize=(8, 8))
c = plt.contour(X0, X1, Y, levels=[5, 10, 15], linestyles='--')
plt.clabel(c, fontsize=10, colors='k', fmt='%.1f')
# plt.plot( [-5, 5], [0,0], '--b')

# plt.plot( [0,0], [-5, 5], '--b')

plt.plot(x_history[:,0], x_history[:,1], 'o')

# plt.xlim(-6, 6)

# plt.ylim(-6, 6)

plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```

![24](https://img.imgdb.cn/item/6031c16b5f4313ce25e9a72c.png)

#### 学习算法的实现

神经网络的学习步骤如下所示。

**前提**

神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的过程称 为“学习”。神经网络的学习分成下面 4 个步骤。

**步骤 1 (mini-batch)**

从训练数据中随机选出一部分数据，这部分数据称为 mini-batch。我们的目标是减小 mini-batch 的损失函数的值。

**步骤 2 (计算梯度)**

为了减小 mini-batch 的损失函数的值，需要求出各个权重参数的梯度。梯度表示损失函数的值减小最多的方向。

**步骤 3（更新参数）**

将权重参数沿梯度方向进行微小更新。

**步骤 4 (重复)**

重复步骤 1、步骤 2、步骤 3。

神经网络的学习按照上面 4 个步骤 进行。这个方法通过梯度下降法更新参数，不过因 为这里使用的数据是随机选择的mini batch 数据，所以又称为**随机梯度下降法**（stochastic gradient descent）。“随机”指的是“随机选择的”的意思，因此，随机梯度下降法是“对随机选择的数据进行的梯度下降法”。深度学习的很多框架中，随机梯度下降法一般由一个名为 `SGD` 的函数来实现。

下面，我们来实现手写数字识别的神经网络。这里以 2 层神经网络（隐藏层为 1 层的网络）为对象，使用 MNIST 数据集进行学习。

##### 2层神经网络的类

```python
# sigmoid(), softmax(), cross_entropy(), numerical_gradient()在前面已经定义过了

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        # 初始化权重
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    # x:输入数据，t: label
    
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
```

| 变量                                                    | 说明                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| `params`                                                | 保存神经网络的参数的字典型变量（实例变量）。<br />params['W1'] 是第 1 层的权重，params['b1'] 是第 1 层的偏置。<br />params['W2'] 是第 2 层的权重，params['b2'] 是第 2 层的偏置。 |
| `grads`                                                 | 保存梯度的字典型变量 (numerical_gradient() 方法的返回值）。<br />grads ['W1'] 是第 1 层权重的梯度，grads['b1'] 是第 1 层偏置的梯度。<br />grads ['W2'] 是第 2 层权重的梯度，grads['b2'] 是第 2 层偏置的梯度。 |
|                                                         |                                                              |
| **方法**                                                | **说明**                                                     |
| `__init__ `(self, input_size, hidden_size, output_size) | 进行初始化。<br />参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数 |
| `predict`(self, x)                                      | 进行识别 (推理)。<br />参数 x 是图像数据                     |
| `loss`(self, x, t)                                      | 计算损失函数的值。<br />参数 x 是图像数据，t 是正确解标签（后面 3 个方法的参数也一样） |
| `accuracy`(self, x, t)                                  | 计算识别精度                                                 |
| `numerical_gradient`(self, x, t)                        | 计算权重参数的梯度                                           |

TwoLayerNet 类有 params 和 grads 两个字典型实例变量。params 变量中保存了权 重参数, 比如 params ['W1'] 以 NumPy 数组的形式保存了第 1 层的权重参数。此外，第 1 层的偏置可以通过 param['b1'] 进行访问。这里来看一个例子。

![27](https://img.imgdb.cn/item/6031c1d35f4313ce25e9cf03.png)

如上所示，params 变量中保存了该神经网络所需的全部参数。并且, params 变量 中保存的权重参数会用在推理处理（前向处理）中。顺便说一下，推理处理的实现如下所示。

![28](https://img.imgdb.cn/item/6031c1d35f4313ce25e9cf08.png)

此外，与 params 变量对应，grads 变量中保存了各个参数的梯度。如下所示，使用 numerical_gradient ( ) 方法计算梯度后，梯度的信息将保存在 grads 变量中。

![29](https://img.imgdb.cn/item/6031c1d35f4313ce25e9cf0e.png)

接着，我们来看一下 TwoLayerNet 的方法的实现。首先是 `__init__` ​(self, input_size, hidden_size, output_size) 方法，它是类的初始化方法（所谓初始化 方法，就是生成 TwoLayerNet 实例时被调用的方法）。从第 1 个参数开始，依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数。另外，因为进行手写数字识别时，输入图像的大小是 $784(28 \times 28)$ ，输出为 10 个类别，所以指定参数input_size=784、output_size=10, 将隐藏层的个数 hidden_size 设置为一个合适的值即可。

此外，这个初始化方法会对权重参数进行初始化。**如何设置权重参数的初始值这个问 题是关系到神经网络能否成功学习的重要问题**。后面我们会详细讨论权重参数的初始化， 这里只需要知道，权重使用符合高斯分布的随机数进行初始化，偏置使用 0 进行初始化。 `predict` (self, x)​ 和 `accuracy`(self, x, t)​ 的实现和上一章的神经网络的推理处理基本一样。另外, `loss`(self, x, t)​ 是计算损失函数值的方法。这个方法会基于 `predict`() 的结果和正确解标签，计算交叉熵误差。

剩下的 `numerical_gradient` (self,  x, t)​ 方法会计算各个参数的梯度。根据数值微分，计算各个参数相对于损失函数的梯度。

> 之后，我们会介绍一个高速计算梯度的方法，称为**误差反向传播法**。用误差反向传播法求到的梯度和数值微分的结果基本一致，但可以高速地进行处理。

##### 基于测试数据的评价

这里先介绍一个概念，叫做**epoch**。epoch 是一个单位。一个 epoch 表示学习中所有训练数据均被使用过一次时的更新次数。比如，对于 10000 笔训练数据，用大小为 100 笔数据的 mini-batch 进行学习时，重复随机梯度下降法 100 次，所有的训练数据就都被“看过”了“。此时, 100 次就是一个 epoch。实际上，一般做法是事先将所有训练数据随机打乱，然后按指定的批次大小，按序生成 mini-batch。这样每个 mini-batch 均有一个索引号，比如此例可以是 $0,1,2, \ldots, 99,$ 然启用索引号可以遍历所有的 mini-batch。遍历一次所有 数据，就称为一个epoch。请注意，之后的代码在抽取mini-batch， 每次都是随机选择的，所以不一定每个数据都会被看到。 

之前提到过，神经网络学习的目标就是要掌握泛化能力。训练数据的损失函数值减小，虽说是神经网络的学习正常进行的一个信号，但光看这个结果还不能说明该神经网络在其他数据集上也一定能有同等程度的表现。因此，要评价神经网络的泛化能力，就必须使用不包含在训练数据中的数据。下面的代码在进行学习的过程中，会定期地对训练数据和测试数据记录识别精度。这里，每经过一个 epoch，我们都会记录下训练数据和测试数据的识别精度。

```python
# 这是导入数据的操作，需要用到load_mnist(), 这个函数在本书提供的代码中有

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 超参数的设定

iters_num = 10000  
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # 更新参数
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 计算每个epoch的识别精度
    
    if i % iter_per_epoch == 0:
        train_acc = network.acuracy(x_train, t_train)
        test_acc = network.acuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
```

在上面的例子中，每经过一个 epoch，就对所有的训练数据和测试数据计算识别精度，并记录结果。之所以要计算每一个 epoch 的识别精度，是因为如果在 for 语句的循 环中一直计算识别精度，会花费太多时间。并且，也没有必要那么频繁地记录识别精度 (只要从大方向上大致把握识别精度的推移就可以了) 。 因此，我们才会每经过一个 epoch 就记录一次训练数据的识别精度。

下一篇就介绍误差反向传播算法。

### 代码汇总

[Deep_Learning_from_scratch_2](https://github.com/leyuanheart/Deep_Learning_From_Scratch/tree/main/2)

---

**例行申明**：博客里大部分的文字和图片会使用这本书中的内容，如果是我自己的一些想法，我会说明的（是不是感觉和一般的申明反过来了...，这是我自己的问题，我是这个年三十决定开始写的，但是之前在读的时候并没有做过笔记，现在手边只有这本书，而且我也不想花太多时间在这上面，毕竟还要写论文呢...，但是又不想让之前看过的这本书就这么过去了，所以就决定这么办，如果涉及到版权问题，我到时候就把文章撤了，在这里先道个歉）。