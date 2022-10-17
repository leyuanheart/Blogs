---
layout:     post
title:      Github Pages + Jekyll 搭建个人博客主页
subtitle:   
date:       2020-06-06
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Github Pages
    - Jekyll
---

# 前言

花了一天时间终于把自己的博客主页给搭建好了，下面就分享一下这个过程。

话不多说，还是先上自己参考过的文章

- [如何搭建独立博客——简明GitHub Pages与Jekyll教程](https://www.kongqigongying.com/blog/2019/05/01/howto-setup-blog/)
- [搭建个人博客教程(基于github pages和jekyll)](https://wangpei.ink/2019/04/21/%E6%90%AD%E5%BB%BA%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2%E6%95%99%E7%A8%8B(%E5%9F%BA%E4%BA%8Egithub-pages%E5%92%8Cjekyll)/)
- [博客搭建详细教程](https://github.com/qiubaiying/qiubaiying.github.io/wiki/%E5%8D%9A%E5%AE%A2%E6%90%AD%E5%BB%BA%E8%AF%A6%E7%BB%86%E6%95%99%E7%A8%8B)
- [GitHub Pages + Jekyll 创建个人博客](https://www.jianshu.com/p/9535334ffd54)
- [采用Jekyll + github 构建个人博客](https://www.jianshu.com/p/32af878fdf69)（这篇主要是安装Jekyll和本地运行时可能会出现的一些问题和解决办法）
- [Git 忽略提交 .gitignore](https://www.jianshu.com/p/74bd0ceb6182)

如果大家看了这几篇文章或者自己搜索了一些这方面的文章，应该都能够明白该怎么做（一定要有耐心，一篇看不太明白，多看几篇，他们里面相似的内容就基本上是有用且正确的了）。我这里归纳整理了一套理解版本，希望能帮助到大家对这个问题形成一个系统的认识。

# Github Pages

如果不知道[Github](https://github.com/)，可以先去了解一下再回来。

[Github Pages](https://pages.github.com/)是一个静态网站托管服务（相当于提供了一个存放网站的服务器），是Github提供的为个人或者是项目创建静态网站的工具，但如果你使用过Github，那就很简单了，其实就把它理解成一个repository（仓库）就行了，只是这个仓库有自己特殊的命名方式和其他的一些操作，接下来就是搬运官网里的介绍。

![image-1](https://pic.downk.cc/item/5f095e6514195aa5942e34a3.png)

新建一个仓库，然后命名成`github用户名.github.io`

![image-2](https://pic.downk.cc/item/5f095e6514195aa5942e34a7.png)

然后就可以在仓库里新建一个`index.html`文件，输入上图的代码，这是HTML语言，用来写网站的。

其实这之前还有一步操作，它会让你选择你是用什么哪种git的方式，如果要涉及到本地操作的话，就要用到Git，如果不熟悉在终端用Git的话，推荐用[GitHub Desktop](https://desktop.github.com/)，同样能很方便的管理代码，而且省去了学git的时间（不过其实也还是要了解git的工作原理的，比如我，就现在[Git 教程-廖雪峰](https://www.liaoxuefeng.com/wiki/896043488029600)学了一遍（很早以前了），然后就选择了Github Desktop......），但其实你在Github网站上也是可以新建、删除、修改文件的，只是不那么方便而已，而且由于在国内，GitHub的速度有时候很感人，要是上传文件，你可能要等到天荒地老了，这里就不讨论这些问题了。

![image-3](https://pic.downk.cc/item/5f095e6514195aa5942e34a9.png)

然后其实你就已经完成了你的网站的建立了，在浏览器里输入`username.github.io`，你就能看到`index.html`文件解析成网页的结果，长成这样。

![image-4](https://pic.downk.cc/item/5f095e6514195aa5942e34ad.png)

如果你熟悉HTML语言的话，其实你直接写就可以了，但是我并不会，所以Jekyll就登场了。

# Jekyll

> Jekyll 是一个简单的博客形态的静态站点生产机器。它有一个模版目录，其中包含原始文本格式的文档，通过一个转换器（如 [Markdown](http://daringfireball.net/projects/markdown/)）和我们的 [Liquid](https://github.com/Shopify/liquid/wiki) 渲染器转化成一个完整的可发布的静态网站，你可以发布在任何你喜爱的服务器上。Jekyll 也可以运行在 [GitHub Page](http://pages.github.com/) 上，也就是说，你可以使用 GitHub 的服务来搭建你的项目页面、博客或者网站，而且是**完全免费**的。

这是[中文版官网](http://jekyllcn.com/)上的介绍，简单来说，Jekyll就是一种将纯文本转换成静态网站的工具(软件)，就相当于它已经把一个网站的框架给你搭好了，比如这块是标题，那块是个人介绍等等，不用你自己去写一些HTML的语言了。你只需要专注写你的博客文，Jekyll负责把它们展示出来。（当然你还是得学习Jekyll的模板目录每块是用来放什么，不然也是白搭）

接下来我们暂时忘记Github Pages，而是专注于Jekyll，首先要面对的就是安装它。

## 安装

虽然 Windows 并不是 Jekyll 官方支持的平台，但是也可以通过合适的方法使其运行在 Windows 平台上。我是参考官网上的安装方式 [官方Windows环境下Jekyll的安装](http://jekyll-windows.juthilo.com/)

### 1. Install Ruby and the Ruby DevKit

Ruby是Jekyll的编程语言，所以安装Jekyll之前需要先安装Ruby和相应的DevKit。

这一步就和我前面提到的一些文章里说的不一样了，应该是因为要么提到安装的就是直接贴官方文档，要么就是2017甚至2016年的文章了，里面Ruby和DevKit是分开安装的，但是现在是可以打包一起安装的，问题不大，按照官网的步骤走，反正我是没有碰到什么问题......

![image-5](https://pic.downk.cc/item/5f095e6514195aa5942e34af.png)

我下载的就是`WITH DEVKIT`里面的第三个，因为右边说了，不确定下哪个的话就下这个，稳当，然后就是正常的解压缩安装了，哦对了，别忘了在安装的时候勾选把工作目录添加到环境变量里的选项，这个看官网的步骤就行了。

### 2. Install the Jekyll Gem

Jekyll本身就是一个用Ruby写的software package，所以可以通过Ruby的`gem`命令来安装

```bash
gem install jekyll
```

到这里基本上就安装完成，你可以通过`jekyll -v`来查看版本号，如果碰到问题，可以参考这两篇文章

- [GitHub Pages + Jekyll 创建个人博客](https://www.jianshu.com/p/9535334ffd54)
- [采用Jekyll + github 构建个人博客](https://www.jianshu.com/p/32af878fdf69)

虽然都是老文了，但仍然是很有用的，安装的时候我没碰到问题，到后面本地启动服务（不是这下面，而是在[在本地启动Jekyll服务来预览博客](#3-在本地启动jekyll服务来预览博客)）的时候还是遇到了一点麻烦，靠它们成功克服了。

## 本地启动服务

官方网站上提供了一个快速启动的例子

```bash
# 安装bundler，bundler通过Gemfile文件来管理gem包
gem install  bundler

# 创建一个新的Jekyll项目，并命名为myblog
jekyll new myblog

# 进入myblog目录
cd myblog

# 创建本地服务器，默认的运行地址为http://localhost:4000
# bundle exec 表示在当前项目依赖的上下文环境中执行命令 jekyll serve
bundle exec jekyll serve
```

`jekyll new myblog`这个过程时间还不短，不要急，等它跑完

![quick start for jekyll](https://pic.downk.cc/item/5f095f1014195aa5942e671b.png)

`cd`到`myblog`目录下，可以看到里面有这些文件

![image-6](https://pic.downk.cc/item/5f095f2414195aa5942e6d22.png)

这和官网上贴出来的以及很多文章里展示的都不太一样，官网是长这个样子

![image-7](https://pic.downk.cc/item/5f095f2414195aa5942e6d24.png)

但是没有关系，这一小部分已经够生成一个看起来还不错的网站了，再运行`bundle exec jekyll serve`，当然直接运行`jekyll serve`也没问题，在浏览器里输入地址，可以看到

![image-8](https://pic.downk.cc/item/5f095f2414195aa5942e6d28.png)

点进去是下面这样的

![image-9](https://pic.downk.cc/item/5f095f2414195aa5942e6d2c.png)

到这里其实已经可以算是入门了Jekyll了，后面的一些基本用法，比如`jekyll build`，`jekyll build --destination <destination>`等，如果你不是单独使用Jekyll的话可以不用掌握，也不要被这个吓着了，后面搭配GitHub Pages使用完全可以忽略这部分。关于目录结构里的每个文件夹对应放什么内容，每个文件表示什么意思，可以查看官网介绍，其他的可能不是深入了解的都不会用到（反正我是这样的）。

![image-15](https://pic.downk.cc/item/5f095f7914195aa5942e8cd2.png)

# GitHub Pages + Jekyll

前面我们介绍了，GitHub Pages提供了一个静态网站的托管服务，Jekyll可以让那些不是很熟悉HTML的人也可以写出自己的网站，而Github Pages 支持自动利用 Jekyll 生成站点，这样 GitHub Pages + Jekyll就能方便地让你自己写的网站被全世界看到了。

下面就是最重要的内容了，如何可以完整的在GitHub上创建自己的博客主页。

## 1. 创建`username.github.io`

按照[GitHub Pages](#github-pages)中的步骤创建自己的`.github.io`仓库，然后用GitHub Desktop克隆到本地。

## 2. 在GitHub上找一个博客模板

![image-10](https://pic.downk.cc/item/5f095f2414195aa5942e6d2f.png)

这上面有很多模板，我用的是第一个（其实我并不是直接用的第一个，而是根据我前面文章里的步骤拷贝了一个模板，之后在总结的时候又找到了它的最原始出处[Huxpro](https://github.com/Huxpro/huxpro.github.io)，不过这个博客主页因为是他自己的主页，经常会更新，所以他提供了一个稳定版 [huxblog-boilerplate](https://github.com/Huxpro/huxblog-boilerplate)，说白了就是供人下载的版本），这是他模板的样子。

![image-11](https://pic.downk.cc/item/5f095f9814195aa5942e977a.png)

你把它下载下来，把解压的文件全部复制到自己的本地仓库里，然后你读他的中文版的[README](https://github.com/Huxpro/huxblog-boilerplate/blob/master/README.zh.md)，基本就知道怎么改它里面的一些设置，然后再结合在本地预览博客，就可以开始玩自己的博客，最主要的两个我放图在这里，其他的就不再介绍了。

![image-12](https://pic.downk.cc/item/5f095f9814195aa5942e977d.png)

![image-13](https://pic.downk.cc/item/5f095f9814195aa5942e9782.png)

## 3. 在本地启动Jekyll服务来预览博客

修改之后你如果想要先预览一下再commit并同步到你GitHub的远程仓库上，就需要在本地启动Jekyll服务了，按照[本地启动服务](#本地启动服务)这一节的步骤，先`cd`到你的博客目录里，然后运行`jekyll serve`，在浏览器中输入地址就可以查看了，比你修改完传到远端仓库再查看效率高很多。

但其实本地运行有时是会出现错误的，最主要的原因是Jekyll的版本不一样导致的问题，因为很多模板并不是及时更新的，由于我并不是直接以这个模板来运行的，所以我不确定会不会有问题，但是我在运行的时候是出现了问题的，原因是模板使用的Jekyll的版本和我安装的版本不一样，所以会出现如下警告

![image-14](https://pic.downk.cc/item/5f095f9814195aa5942e978b.png)

你出现的警告可能和这个不完全一样，但类型应该是一样的，都是`unexpected character`，这时候你要么就是卸掉你安装的最新版的Jekyll，安装它指定版本的

```bash
gem uninstall jekyll -v 你自己的版本
gem install jekyll -v 模板的版本（如果你知道的话，一般这个信息可以在Gemfile里看到，但是这个模板并没有...）
```

要么就是按照错误中的指示，在`_layouts`下的`post.html`和`page.html`中

- 将`&&`替换成`and`（这一步可能不需要，你要根据错误里面写的来）
- 将\{\{site.featured-condition-size\}\}替换成`site.featured-condition-size`（也就是删除中括号）

如果还有什么其他的问题出现，就参考我之前提到的两篇博客

- [GitHub Pages + Jekyll 创建个人博客](https://www.jianshu.com/p/9535334ffd54)
- [采用Jekyll + github 构建个人博客](https://www.jianshu.com/p/32af878fdf69)

每次你改完了，可以本地刷新浏览器看结果（要多刷几遍，或者等一会再刷），没问题了再同步到远端仓库，就OK了，最后再分享一些使用GitHub Pages + Jekyll的注意事项

- `_posts`是放置你博客文章的地方，命名规则很重要，必须要符合: `YEAR-MONTH-DAY-title.md` ，并且不能有空格，只能用英文，反正名字可以在博客文的YAML里再指定，所以这里就遵守规则就好

- 一般GitHub项目里都会有一个`.gitignore`的文件，是用来忽略一些我们不需要提交到远程仓库的更新，比如`__pycache__`，以及这里会涉及到的`_site`，最好再加上在本地启动服务会产生的`.jekyll-cache`文件夹，文件里可以这么写

  ```bash
  _site
  .jekyll-cache
  ```

  详细了解可以参考[Git 忽略提交 .gitignore](https://www.jianshu.com/p/74bd0ceb6182)

- markdown里插入图片在本地是可以指定路径的，但是如果要放到GitHub上，那对应的图片也需要上传，GitHub Pages虽然是免费的，但也不是无限制的，比如

  - 仓库空间不大于1G

  - 每个月的流量不超过100G

  - 每小时更新不超过 10 次

    所以如果博文多的话，其实还是有问题的，这个时候你就需要**图床**这个东西了，至于什么是图床，我之前提供的文章里也有介绍，有需要的可以再仔细看看

    

    

