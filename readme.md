**AI大语言模型学习笔记**
![background](images/suse.jpg)

# **0. 前言**
我写这个笔记用来记录我最近几个月学习大语言模型的过程。

写的代码基本都是看网上的资料学习的，原创的代码内容很少。

感谢各位大佬的开源资料。

主要学习资料就是github上的[THUDM](https://github.com/THUDM)

学习大预言模型不仅仅是简单调用，还包括按私有化场景开发垂直应用。针对某个具体的领域，深挖AI的潜力。

# **1.准备环境**
## **1.0 硬件基础**
### **显卡：3070以上。**
大语言模型主要依赖英伟达的CUDA计算模块，使用CPU等可以进行模拟和运行，但仅限于可以跑通，但运行消耗的时间超过GPU运行的7-10倍以上，而且非常消耗内存（20G起）。所以如果想要本地运行大语言模型，推荐使用3070以上的英伟达显卡。我这些程序在我的3070、3080、4090显卡、在公司A40显卡，可以正常运行。
#### **内存：32Gb以上**
虽然运算主要跑在GPU，但我发现其实内存占用也挺大的，我2台32G 1台64G内存的电脑，经常跑起来会占用内存超过16G，推荐有条件最好有32G以上内存。
#### **CPU: 主流即可**
我几台电脑CPU是 5900HX，11800H，13900KF 都能顺利运行，整个训练和推理不怎么依赖CPU，主流即可。
## **1.1 软件基础**
#### **操作系统：Windows 11.**
网上的教程都是推荐使用Linux ubuntu. 我个人是很喜欢Linux的，但大部分用户电脑不可能为了学习大语言模型额外安装一个操作系统，所以我尽量在windows环境下完成这些AI的开发、训练和应用。有关脚本(.sh)和程序我都修改为可以在windows环境下直接运行的了。

为了安装deepspeed，我拿一台电脑也尝试安装了Ubuntu,结果配置环境问题也很多，最终放弃。考虑过虚拟机安装Linux，但网上资料似乎说虚拟后显卡性能打折不少，就放弃了虚拟方案。
#### **英伟达显卡最新驱动和CUDA:**
[英伟达驱动下载](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

我写笔记的时候，显卡驱动版本是
![nvidia-smi](images/nvidia_smi.png)


#### **Python环境：**
请安装Python 3.10.11  [下载地址](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

#### **Pycharm Python IDE:**
当然不用ide，直接通过powerShell命令行来运行是可以的，但我觉得还是pycharm方便一点。推荐直接使用jetbrains出品的toolbox来安装和管理IDE。

安装后启动toolbox，选择pycharm community社区版安装即可。

[下载地址](https://www.jetbrains.com/zh-cn/toolbox-app/download/download-thanks.html?platform=windows)

Git安装

[Git下载地址](https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe)

如果不习惯用命令行，可以用 git GUI的 乌龟git

[乌龟git下载链接](https://download.tortoisegit.org/tgit/2.17.0.0/TortoiseGit-2.17.0.2-64bit.msi)

## **1.2 心理准备**
学习大语言模型训练比我之前学习其他的IT技术，都要更加花费时间和精力。过程中经常会报各种错，查很多资料都未必能解决。

我这个笔记会把从0开始搭建环境讲一次，如果遇到不可解决的困难，可以直接下载我打包好的文件，解压导入pycharm，设置编译器即可运行，简单粗暴。

希望大家要努力，不要搞成《大语言模型学习从入门到放弃》。

# **2.ChatGLM本地部署**

从这个章节开始，我会正常讲从0开始怎么拉取代码和配置环境。
如果你觉得这些太困难，可以直接从2.3开始看，直接百度盘下载配置后的结果，可以快速上手。

## **2.1 本地部署（复杂版)**

需要访问github拉取代码。
如果访问github有困难的，建议安装fastgithub。[百度盘下载FastGithub链接](https://pan.baidu.com/s/1iKMY0w1Y7sDW-aQx2LA4Mw?pwd=nn6c)

下载得到一个文件夹，打开运行 FastGithub.UI 即可。
![fastgithub](images/fastGithub.png)

### **2.1.1 下载代码**
创建一个文件夹，放代码。

`git clone https://github.com/baidongyi/xbbGLM.git`

创建一个虚拟python环境。
![创建虚拟的Python环境](images/python%20venv.png)

### **2.1.2 下载model**

从huggingface科学上网下载
[下载](https://huggingface.co/THUDM/chatglm-6b)

从我的百度云盘下载
[下载](https://pan.baidu.com/s/15GVjxkSNRHmCOFfkRpkjSA?pwd=qqkg)

下载以后得到:
![模型文件](images/model.png)

一定要下载完整，不能少文件。

### **2.1.3 安装依赖**

需要使用pip安装大量包，可考虑设置使用TH的下载源，提高速度

`pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`


`pip install -r ./requirements.txt`

requirements里没有torch，这个需要手动安装cuda版本。默认pip会很容易安装cpu版本。然后运行起来很慢，吃很多内存，然后GPU显存不怎么用。

请根据 [torch下载链接](https://pytorch.org/get-started/locally/)下载安装，文件比较大，下载耗时间。
![pytorch下载界面](images/pytorch.png)
安装之后，运行以下代码，确保成功安装了torch cuda版本

`python`

`import torch`

`print(torch.cuda.is_available())`

![torch cuda](images/cuda_torch.png)

返回True表示安装cuda torch成功。False则表示失败。

### **2.1.4 运行测试代码**

运行测试代码及查看结果。
![运行测试代码](images/run1.png)


## **2.2 本地部署（简化版）直接使用我的文件**

如果已完成2.1的内容，则可以跳过本节2.2内容，直接看2.3的微调ptune。

如果实在用不惯git以及网络受限，可以直接下载我百度盘的资料，解压即可。

### **2.2.1 下载文件**

[百度网盘](https://pan.baidu.com/s/1KWCAkOSgCg64LoONUt38-g?pwd=2awp)

下载后放到一个文件夹。
![文件路径](images/path.png)

### **2.2.2 创建python虚拟环境并覆盖lib文件夹**

在pycharm里创建好虚拟环境 .venv。然后把lib.zip解压放到虚拟环境文件夹.venv里。这样很多依赖就已经安装了。

在pycharm里打开项目文件夹，在setting里创建新的虚拟环境.venv

把百度网盘里下载的lib.zip解压后放到 新创建的.venv虚拟环境文件夹内 覆盖 Lib文件夹

![覆盖.venv里的lib](images/lib.png)

等Pycharm扫描完成后（打开pycharm等待它扫描installed packages），可以看到发现了这些已安装的包。

![](images/package.png)

### **2.2.3 修改配置文件**
打开XT00_parameter.py 编辑一些配置。很多地方要用到这个路径和配置，我集中放在这个文件里统一编辑了，免得用户导出修改路径。

修改model\_path 为 chatGLM-6b-int ，百度盘下载的model.这个model的授权为原作者规定的Licence，本笔记仅做学习记录之用。如果有侵权，请通知我立刻删除。我十分尊重原作者知识产权。

记得在绝对路径前面加个r，免得路径里所有\都要改为\\转义。

![修改参数](images/para.png)

### **2.2.4 测试运行API**
#### **2.2.4.1 运行API**

![启动API](images/api_run.png)


#### **2.2.4.2 验证API**

![验证API](images/api_call.png)

收到GLM的回复，表示运行完成。

### **2.2.5 测试运行CLI**
也可以通过命令行CLI来跟GLM对话

![运行cli](images/cli_run.png)


### **2.2.5 GUI Gradio验证** ###

喜欢用GUI的，也可以通过Gradio快速构建用户界面。

可以安装 

`pip install gradio==5.12.0`

运行图形界面
![运行图形界面](images/gui_run.png)

使用图形界面
![使用图形界面](images/gradio_show.png)




## **2.3 ptune v2微调**
### **2.3.1 ptune微调是什么**

看看这个语言模型自己怎么说的。

![ptune是什么](images/what%20is%20ptune.png)

### **2.3.2 微调的几个步骤**

0. 设置训练的参数
1. 转换资料为训练的格式
2. 开始训练
3. 进行训练效果评估
4. 使用微调后的模型


### **2.3.3 微调的参数**

开始修改代码前，先检查文件路径

![文件路径](images/path.png)

![main文件夹内](images/main_path.jpg)

修改路径
![修改路径](images/para.png)

### **2.3.4 转化数据格式**
官方给的训练数据集是json格式，我写了程序可以帮大家把excel格式的一问一答格式转换为训练直接使用的json格式。

Excel文件格式：ptune/source_files/base.xlsx

![Excel问题库格式](images/excel.png)


Q列放问题，A列放回答。可以放多个文件xlsx,格式保持这个即可。我的程序会一起转换。转换后的文件放在：
ptuning/dest_files 文件夹。

![微调修改路径](images/ptune_para.png)

运行转化
![运行转化文件](images/convert.png)

运行转化结果
![运行转化文件结果](images/convert2.png)

运行结束，查看转化好的训练文件
![转化好的训练文件](images/train_file.png)


### **2.3.5 进行微调训练**

开始进行训练,查看过程和进度：一般需要3-5个小时。
![运行训练](images/run_train.png)

训练完成后，生成的文件路径，我们设置一共训练3000steps,每1000steps保存一次，所以有3个文件夹checkpoints。
![训练结果文件](images/train_result.png)


### **2.3.6 评估微调后的模型效果**
运行评估效果
![进行评估效果](images/ptune_eval.png)

简单解释一下这些指标：
![bleu](images/bleu.png)


### **2.3.7 使用微调后的模型**

使用微调后的模型。

![使用模型](images/use_trained.png)

我的训练效果不好，我训练了很多次。

![多次训练结果](images/more_train.png)


# **3.LangChain本地部署**
## **3.1 本地文件资料搜索**
## **3.2 与ChatGLM集成，实现基于本地知识库问答**


