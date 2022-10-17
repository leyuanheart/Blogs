---

layout:     post
title:      在服务器上配置Jupyter并远程登录
subtitle:   从conda安装到Jupyter配置简明教程
date:       2021-03-16
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Linux
    - conda
    - Jupyter

---

# 前言

这一篇来介绍一下如何在服务器上配置Jupyter并且在本地远程登录使用。比起直接连接服务器出现的terminal，很多人还是更熟悉有目录结构的界面，比如这样：

![1](https://img.imgdb.cn/item/605013635aedab222cfa21f9.png)

反正连服务器大部分时间使用来程序的，如果还用的是python的话，那再顺手配置一下Jupyter绝对不吃亏。

## Linux上安装conda

安装python的话建议直接安装conda，里面集成了python，而且最主要的好处是conda是一个软件包管理系统和环境管理系统，可以解决很多软件包安装过程中的依赖问题，还可以很轻松地创建多个互不影响的虚拟环境。

然后我再推荐安装miniconda而不是Anaconda。两者都是conda的发行版，前者相当于是后者的一个精装版，大小超不多是10:1的关系。我们主要是需要conda和python（甚至python都不一定用，有了conda之后就可以自己创建带有指定版本python的虚拟环境），而miniconda里都有，其他一些科学计算的工具其实用不太到，对于服务器来说，尽量精简安装比较好，留足存储空间给更重要的内容。

下载地址我放在下面，有官方的和清华镜像的（下载速度更快）：

- [官方miniconda](https://repo.anaconda.com/miniconda/)

- [清华镜像miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)

- [官方Anaconda](https://repo.anaconda.com/archive/)

- [清华镜像Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

Anaconda版本和python版本的对应关系参考网站[Old package lists](https://docs.anaconda.com/anaconda/packages/oldpkglists/)，minconda安装文件名上就有对应的python版本。下面我以miniconda为例，Anaconda下载安装基本是一样的。

可以直接点击下载，也可以使用命令下载：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-MacOSX-x86_64.sh
```

下载完成后使用命令安装：

```bash
bash Miniconda3-py37_4.9.2-MacOSX-x86_64.sh   # 安装上面下载的文件
```

然后根据提示按enter键或者输入yes，即可。

最后安装成功之后，会在家目录`~`下生成一个`minconda3`文件夹

此时还不能调用`conda`命令，还需要激活安装，有点类似重启的意思。使用命令：

```bash
source ~/.bashrc   # 刷新一下环境变量
```

然后应该就OK了，可以输入`conda --version`试一下，如果能输出版本信息，就说明安装成功了。

如果要卸载miniconda，只需要把`minconda3`文件夹用`rm`命令删除，然后用`vim`命令打开`.bashrc`文件，将里面和conda有关的语句注释掉，然后重新激活一下`source ~/.bashrc`.

## conda切换国内源

可以命令行中直接输入以下命令：

清华源

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# 如果要安装 pytorch还要添加下面的源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

中科大源：

```bash
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/ 
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/ 
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/ 
conda config --set show_channel_urls yes
```

如果是在Linux系统上，也可以将以上配置文件写在`~/.condarc`中，先输入命令`vim ~/.condarc`，然后编辑该文件

```bash
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

在修改完成之后，一定要重新启动一个新的Shell，否则设置不生效。

如果要切换回默认镜像源，命令行输入：

```bash
conda config --remove-key channels
```

关于`pip`换源的方法，一般都是临时使用，就是在命令后面加上参数`-i 国内源`，比如

```bash
pip install some-package -i https://mirrors.aliyun.com/pypi/simple/
```

一些常见的国内源：

```bash
# 豆瓣
https://pypi.doubanio.com/simple/
# 阿里云    
https://mirrors.aliyun.com/pypi/simple/
# 清华大学
https://pypi.tuna.tsinghua.edu.cn/simple/
https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

如果要设为默认的话，升级 `pip `到最新的版本后进行配置：

```bash
pip install pip -U
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## conda创建虚拟环境

这其实是我们安装conda最主要的目的。我们为啥需要虚拟环境？原因是不同项目所依赖的包的版本是不一定相同（甚至可以说肯定不相同）。只在同一个环境中安装和更新包，新安装的版本会覆盖以前的版本，那么就无法实现同时去处理多个项目。因此就需要虚拟环境，它可以搭建独立的python运行环境，这样每个项目有自己的环境，彼此不会相互影响。

### 创建虚拟环境

比如你想创建一个名为`py3.6`的使用Python版本为3.6的环境，你可以在命令行输入：

```bash
conda create -n py3.6 python=3.6
```

输入命令后，会在miniconda安装目录下的`envs`目录里找到这个环境

你可以使用`conda env list`来查看当前存在哪些虚拟环境

### 切换虚拟环境

```bash
conda activate py3.6
```



### 安装额外的包

你可以选择先进入该虚拟环境，然后再按照一般安装包的流程安装你需要的包，也可以指定要安装包的环境：

```bash
conda install -n py3.6 [package]
```

### 关闭虚拟环境

```bash
conda deactivate
```

### 删除虚拟环境

```bash
conda remove -n your_env_name(虚拟环境名称) --all
conda remove --name your_env_name  package_name  # 删除环境中的某个包
```

## 安装jupyter notebook

终于是回到这篇文章的主题了，我使用的是jupyter notebook，也有很多人是使用的它的进化版，jupyter lab。但是对我来说jupyter notebook已经够用了。好像如果是安装了Anaconda，那么已经默认安装了jupyter lab（lab里也包括了notebook）了，可以直接进行配置，可以通过`conda list`查看已经安装了包，如果没装，就先安装一下

```bash
conda install jupyter notebook
# 或者用pip安装
pip install jupyter notebook
```

## 生成配置文件

为了生成配置文件，需要使用下面的jupyter命令

```bash
jupyter notebook --generate-config
```

此时就会得到一个配置文件，其默认路径一般如下所示：

```bash
Windows: C:\Users\USERNAME\.jupyter\jupyter_notebook_config.py
OS X: /Users/USERNAME/.jupyter/jupyter_notebook_config.py
Linux: /home/USERNAME/.jupyter/jupyter_notebook_config.py
```

## 设置登录密码

### 自动设置(推荐)
在jupyter5.0以后的版本，可以使用`jupyter notebook password`来设置密码:

```bash
jupyter notebook password
Enter password:  yourcode  #输入密码
Verify password: yourcodeagain   #再次输入密码确认
#运行后结果
[NotebookPasswordApp] 
Wrote hashed password to /Users/you/.jupyter/jupyter_notebook_config.json    
#密码被保存的位置 ~/.jupyter/jupyter_notebook_config.json
```

### 手动设置

```bash
#利用Ipython工具来设置密码
$ ipython
#进入ipython环境
In [1]: from notebook.auth import passwd    #导入授权模块设置密码
In [2]: passwd()
Enter password:    yourcode          #输入密码
Verify password:    yourcodeagain    #再次输入密码
Out[2]: 'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'    #这是一串密码的哈希值
#将在~/.jupyter/jupyter_notebook_config.py配置文件中设置，需要保存好
```

## 修改配置文件

输入以下命令：`vim ~/.jupyter/jupyter_notebook_config.py`

对于自动模式，打开配置文件后修改三个地方：

```bash
#把前面的#去掉
c.NotebookApp.ip = '*'    #允许所有ip访问  补充:报错 No address associated with hostname可设置为:'0.0.0.0'
c.NotebookApp.open_browser = False    #不打开浏览器
c.NotebookApp.port = 8888             #端口为8888
```

对于手动方法，除了上述修改之外，还需要配置哈希秘钥：

```bash
#配置刚刚生成的秘钥，一长串哈希码
c.NotebookApp.password = u'sha1:bcd259ccf...<your hashed password here>'
```

配置完成后启动jupyter notebook，直接输入命令`jupyter notebook`.

注意：此时启动就可以在局域网内的其它电脑上访问jupyter了，在浏览器输入localhost:8888（电脑本身已经安装了jupyter notebook）就会有登录界面，输入刚才设置的密码即可。

如果需要在外网访问，需要设置端口转发：利用路由器的端口转发功能或者使用花生壳等内网穿透来实现，将这台电脑的端口绑定到对应外网的ip的某个端口上。

主要内容介绍完了，接下来是一些和jupyter相关的平时可能会用的比较多的内容，我也总结一下。

## 如何在jupyter中添加conda虚拟环境

要想在jupyter notebook里使用创建的虚拟环境，需要进行一些设置。

1、先确认一下base环境中有没有安装ipykernel。可以用`conda list`或者`pip list`命令。没有的话就安装一下：

```bash
conda install ipykernel
# 或者
pip install ipykernel
```

2、在创建虚拟环境的时候添加ipykernel，或者创建完之后再安装ipykernel也可以。

```bash
conda create -n env_name python=3.6 ipykernel
```

3、切换到虚拟环境中，并将环境写入notebook的ipykernel中

```bash
conda activate env_name
python -m ipykernel install --user --name env_name --display-name "在jupyter中显示的环境名称"
```

4、如果想要在kernel中删除某个环境

```bash
jupyter kernelspec remove env_name
```

## Jupyter Notebook插件管理与安装

Jupyter Notebook本身有推荐的插件管理包。我们首先需要安装如下第三方库用于管理Jupyter Notebook插件。

```bash
pip install jupyter_nbextensions_configurator
pip install jupyter_contrib_nbextensions
```

安装完成后，我们还需要执行如下命令来完成插件管理启用：

```bash
jupyter nbextensions_configurator enable --user
jupyter contrib nbextension install --user
```

再次打开jupyter notebook，你会发现上方多了一个`Nbextentions`的选项，点进去里面就有很多类型的插件。

![2](https://img.imgdb.cn/item/605013635aedab222cfa21fb.png)

一般用的最多的就是增加目录了，勾选`Table of Contents(2)`。再重启一下jupyter notebook，打开一个notebook，会增加下图中红色圆圈处的一个按钮，点击一下就会出现左侧的目录（前提是你的notebook里有markdown语法的标题）。

![3](https://img.imgdb.cn/item/605013635aedab222cfa21ff.png)

# 参考资料

[linux安装或卸载miniconda](https://www.jianshu.com/p/fab0068a32b4)

[linux 服务器安装 anaconda](https://www.cnblogs.com/zwq-zju/p/9715162.html)

[conda创建python虚拟环境](https://www.cnblogs.com/shierlou-123/p/11138920.html)

[JupyterNotebook配置远程登录](https://blog.csdn.net/u014636245/article/details/83652126?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param)

[Mac、windows环境下使用jupyter远程连接服务器（手把手带你搞定）](https://blog.csdn.net/hahameier/article/details/98874226?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)

[如何在jupyter中添加conda虚拟环境](https://blog.csdn.net/qq_41689620/article/details/95874096)

[JupyterNotebook插件管理与安装](https://www.missshi.cn/api/view/blog/5ab10ef65b925d0d4e000001)

[Jupyter Notebook 插入图片的几种方法](https://blog.csdn.net/zzc15806/article/details/82633865)

[Jupyter Notebook 怎样打开指定文件夹](https://blog.csdn.net/xu380393916/article/details/105637466)