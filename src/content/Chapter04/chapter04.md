# TensorFlow的启动和运行

&emsp;&emsp;在《Deep Learning》一书的第四章中，我们要学习广泛使用的深度学习框架——TensorFlow。目前TensorFlow的社区支持日益增长，这使它成为构建复杂深度学习应用程序的一个不错的选择。下面是从TensorFlow网站上截取的一段话：<br>
&emsp;&emsp;“TensorFlow是使用数据流图进行数值计算的开源软件库。图中的节点表示数学运算，而图边表示它们之间通信的多维数据阵列（张量）。灵活的体系结构允许您使用单个API将计算部署到桌面、服务器或移动设备中的一个或多个CPU或GPU上。TensorFlow最初是由在Google机器智能研究组织内的Google Brain Team工作的研究人员和工程师开发的，用于进行机器学习和深度神经网络研究，但是该系统足够通用，可以广泛地应用于其他具有多样性的各个领域。”<br>
&emsp;&emsp;本章将介绍以下主题：<br>
&emsp;&emsp;（1）TensorFlow的安装；<br>
&emsp;&emsp;（2）TensorFlow的环境；<br>
&emsp;&emsp;（3）计算图；<br>
&emsp;&emsp;（4）数据流类型、变量和占位符；<br>
&emsp;&emsp;（5）从TensorFlow中获取输出；<br>
&emsp;&emsp;（6）TensorBoard的可视化学习。<br>

## TensorFlow的安装
&emsp;&emsp;TensorFlow安装有两种模式：CPU和GPU。首先我们将在GPU模式下安装TensorFlow，开始安装教程。<br>

### Ubuntu 16.04的TensorFlow GPU安装
&emsp;&emsp;因为TensorFlow的GPU版本目前只支持CUDA，所以TensorFlow的GPU模式安装需要最新的NVIDIA驱动程序安装。下面的部分将带您一步一步地安装NVIDIA驱动程序和CUDA 8。<br>

#### 安装NVIDIA驱动程序和CUDA 8
&emsp;&emsp;首先，您需要根据您的GPU安装正确的NVIDIA驱动程序。我（作者）有GeFig GTX 960M GPU，所以我会继续安装nvidia-375（如果你有不同的GPU，你可以使用NVIDIA搜索工具 http://nvidia.com/Download/index.aspx 帮助您找到正确的驱动程序版本)。如果你想知道你的机器的GPU，你可以在终端中发出下面的命令：<br>
`lspci | grep -i nvidia`<br>
&emsp;&emsp;然后您会在终端中获得以下输出：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/GPU%E7%9A%84%E6%9F%A5%E7%9C%8B.png)
&emsp;&emsp;接下来，我们需要添加一个专有的NVIDIA驱动程序库来安装驱动程序。使用apt-get:<br>
`Sudo add-apt-repository ppa:graphics-drivers/ppa`<br>
`Sudo apt-get updata`<br>
`Sudo apt-get install nvidia-375`<br>
&emsp;&emsp;在成功安装NVIDIA驱动程序后，重新启动机器。为了验证驱动程序是否正确安装，在终端中发出以下命令：<br>
`Cat /proc/driver/nvidia/versio`<br>
&emsp;&emsp;然后您会在终端中获得以下输出：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/NVIDIA%E7%9A%84%E5%AE%89%E8%A3%85.png)<br>
&emsp;&emsp;接下来，我们需要安装CUDA 8。打开以下CUDA下载链接：https://developer.nvidia.com/cuda-downloads. 按照以下截图选择操作系统、体系结构、发行版本和安装程序类型：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/CUDA%E7%89%88%E6%9C%AC%E7%9A%84%E9%80%89%E6%8B%A9.png)<br> 
&emsp;&emsp;安装程序文件约为2 GB。您需要发出以下安装说明：<br>
`Sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb`<br>
`Sudo apt-get update`<br>
`Sudo apt-get install cuda`<br>
&emsp;&emsp;接下来，我们需要将库添加到.bashrc文件中，通过发出以下命令：<br>
`echo ‘export PATH=/usr/local/cuda/bin:$PATH’ >> ~/.bashrc`<br>
`echo ‘exportLD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH’>>~/.bashrc`<br>
`Source ~/.bashrc`<br>
&emsp;&emsp;接下来，您需要通过发出以下命令来验证CUDA 8的安装：<br>
`nvcc -v`<br>
&emsp;&emsp;然后您会在终端中获得以下输出：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/CUDA8%E7%9A%84%E5%AE%89%E8%A3%85.png)  
&emsp;&emsp;最后，在本节中，我们需要安装CUDNN 6.0。这个NVIDIA CUDA深神经网络图书馆(CUDNN）是一个GPU加速的深层神经基元库网络。你可以从Nvidia的网页下载。发布以下命令来提取和安装CUDNN：<br>
`cd ~/Downloads/`<br>
`tar xvf cudnn*.tgz`<br>
`cd cuda`<br>
`sudo cp */*.h /usr/local/cuda/include/`<br>
`sudo cp */libcudnn* /usr/local/cuda/lib64/`<br>
`sudo chmod a+r /usr/local/cuda/lib64/libcudnn*`<br>
&emsp;&emsp;为了确保安装成功，可以使用nvidia-smi终端中的工具。如果安装成功，该工具将为您提供诸如RAM之类的监控信息以及GPU的运行过程。<br>

### 安装TensorFlow
&emsp;&emsp;在准备TensorFlow的GPU环境之后，我们现在准备在GPU模式下安装TensorFlow。但是，为了完成TensorFlow的安装过程，您可以首先安装一些有用的Python包，这些包将在下一章中帮助您并使您的开发环境更容易。<br>
&emsp;&emsp;我们可以通过发出以下命令来安装一些数据操作、分析和可视化库：<br>
`sudo apt-get update && apt-get install -y python-numpy python-scipy python-nose python-h5py python-skimage python-matplotib python-pandas python-sklearn python-sympy`<br>
`sudo apt-get clean && sudo apt-get autoremove`<br>
`sudo re -rf /var/lib/apt/lists/*`<br>
&emsp;&emsp;接下来，您可以安装更有用的库，例如虚拟环境、Juyter  Notebook等：<br>
`sudo apt-get update`<br>
`sudo apt-get install git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcur13-dev`<br>
`sudo apt-get install -y libfreetype6-dev libpng12-dev`<br>
`pip3 install -U matplotlib ipython[all] jupyter pandas scikit-image`<br>
&emsp;&emsp;最后，我们可以通过发出以下命令开始在GPU模式下安装TensorFlow:<br>
`pip3 install --upgrade tensorflow-gpu`<br>
&emsp;&emsp;您可以使用Python验证TensorFlow的成功安装：<br>
`import tensorflow as tf`<br>
`a = tf.constant(5)`<br>
`b = tf.constant(6)`<br>
`sess = tf.Session()`<br>
`sess.run(a+b)` #这时应该打印一堆显示设备状态的消息，如果安装过程一切顺利，您应该看到设备中列出的GPU<br>
`sess.close()`<br>
&emsp;&emsp;您应该在终端中获得以下输出：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/tansorflow%E7%9A%84%E5%AE%89%E8%A3%85.png)
### Ubuntu 16.04的TensorFlow CPU安装
&emsp;&emsp;在本节中，我们将安装CPU版本，在安装之前不需要任何驱动程序。因此，让我们开始安装一些有用的数据操作和可视化包。<br>
`sudo apt-get update && apt-get install -y python-numpy python-scipy python-nose python-h5py python-skimage python-matplotib python-pandas python-sklearn python-sympy`<br>
`sudo apt-get clean && sudo apt-get autoremove`<br>
`sudo re -rf /var/lib/apt/lists/*`<br>
&emsp;&emsp;接下来，您可以安装更有用的库，例如虚拟环境、Juyter Notebook等：<br>
`sudo apt-get update`<br>
`sudo apt-get install git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcur13-dev`<br>
`sudo apt-get install -y libfreetype6-dev libpng12-dev`<br>
`pip3 install -U matplotlib ipython[all] jupyter pandas scikit-image`<br>
&emsp;&emsp;最后，您可以通过发出以下命令在CPU模式中安装最新的TensorFlow:<br>
`pip3 install --upgrade tensorflow`<br>
&emsp;&emsp;运行以下命令可以检查TensorFlow是否成功安装TensorFlow语句：<br>
`import tensorflow as tf`<br>
`a = tf.constant(5)`<br>
`b = tf.constant(6)`<br>
`sess = tf.Session()`<br>
`sess.run(a+b)`<br>
`sess.close()`<br>
&emsp;&emsp;您应该在终端中获得以下输出：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/Ubuntu%2016.04%E7%9A%84TensorFlow%20CPU%E5%AE%89%E8%A3%85.png)<br>
### Mac OS X的TensorFlow CPU安装
&emsp;&emsp;在本节中，我们将为Mac OS X安装TensorFlow,使用vortualenv. 那么，让我们从安装QJQ通过发出以下命令的工具：<br>
`sudo easy_install pip`<br>
&emsp;&emsp;接下来，我们需要安装虚拟环境库：<br>
`sudo pip install --upgrade virtualenv`<br>
&emsp;&emsp;在安装虚拟环境库之后，我们需要创建一个容器或虚拟环境，该容器或虚拟环境将托管TensorFlow和您可能想要安装的任何包的安装，而不影响底层主机系统：<br>
`virtualenv --system-site-packages targetDirectory # for Python 2.7`<br>
`virtualenv --system-site-packages -p python3 targetDirectory # for Python3.n`<br>
&emsp;&emsp;这假设targetDirectory是~/tensorflow.<br>
&emsp;&emsp;既然已经创建了虚拟环境，可以通过发出以下命令来访问它：<br>
`source ~/tensorflow/bin/activate`<br>
&emsp;&emsp;发出此命令后，您将访问刚刚创建的虚拟机，并且可以安装仅安装在此环境中且不会影响正在使用的底层或主机系统的任何包。<br>
&emsp;&emsp;为了从环境中退出，可以发出以下命令：<br>
`deactivate`<br>
&emsp;&emsp;请注意，现在我们确实希望处于虚拟环境中，所以现在就把它打开。一旦你完成了TensorFlow的游戏，你就应该停用它：<br>
`source bin/activate`<br>
&emsp;&emsp;为了安装TensorFlow的CPU版本，可以发出以下命令，这些命令还将安装TensorFlow需要的任何依赖库：<br>
`(tensorflow)$ pip install --upgrade tensorflow   #for python 2.7`<br>
`(tensorflow)$ pip3 install --upgrade tensorflow   #for python 3.n`<br>
### Windows的TensorFlow GPU/CPU安装
&emsp;&emsp;我们假设您的系统已经安装了Python 3。若要安装TensorFlow，请按如下方式启动管理员。打开起点菜单，搜索CMD，然后右键点击它并点击作为管理员运行:<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter04/chapter04_images/Windows%E7%9A%84TensorFlow%E5%AE%89%E8%A3%85.png)<br> 
&emsp;&emsp;一旦打开了命令窗口，就可以发出以下命令来在GPU模式中安装TensorFlow:<br>
&emsp;&emsp;[你需要拥有pip或pip3（取决于您的Python版本）在发出下一个命令之前安装。]<br>
`pip3 install --upgrade tensorflow-gpu`<br>
&emsp;&emsp;发出以下命令来在CPU模式下安装TensorFlow:<br>
`pip3 install --upgrade tensorflow`<br>

# 一级标题
## 二级标题
### 三级标题
#### 四级标题
  
  
## 缩进<br>
&emsp;&emsp;这是段落开头的缩进<br>
  
  
## 加粗<br>
这是两个加粗的**字体**
  
  
## 超链接<br>
[百度一下，你就知道](http://www.baidu.com/)
  
  
## 代码  
### 单行代码  

`print('hello world')`  


  
### 多行代码  
```python
for i in range(10):
    print(i, end = '\t')
```  

  
## 插入数学公式，使用在线Latex公式编辑器  

- 百度搜索`在线Latex公式编辑器`，在线编辑好公式后，右击生成的公式图片，复制图片链接地址，以`![](这里粘贴复制的图片链接地址)`格式显示图片。  

![](https://github.com/yanjiusheng2018/dlt/blob/master/image/tupian2.jpg)  

- 插入公式如下：  



![](http://latex.codecogs.com/gif.latex?%5Csqrt%7Ba%5E2&plus;b%5E2%7D)  

![](http://latex.codecogs.com/gif.latex?2H_2%20&plus;%20O_2%20%5Cxrightarrow%7Bn%2Cm%7D2H_2O)  




## 换行  
life is short ,  <br/>you need python!  


  
## 引用  
>一级引用  
>>二级引用  
  
  
## 分割线  
***
---
  
  
## 列表标记  
### 专业  
1. 应用统计  
2. 理统  
3. 概率论  

### 性别  
* 男  
* 女  
  
 ## 图片  
 
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/image/python.jpg)  
   
 ## 表格  
   
   
学号|姓名|专业
-|-|-
1|李|应用统计
2|王|理学统计
3|张|基础数学
<br>

## 注意  

- 文章内需要使用图片时，先在对应章节文件夹内上传图片，然后右击复制图片链接地址，在文章需要使用图片的地方英文状态下输入`![图片描述]（这里粘贴图片链接地址）`格式显示图片。  

- 每次编辑内容后，记得提交保存，如下图所示：  
![点击提交](https://github.com/yanjiusheng2018/dlt/blob/master/image/tupian.jpg)
