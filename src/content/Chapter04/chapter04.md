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
