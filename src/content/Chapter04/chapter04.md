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
