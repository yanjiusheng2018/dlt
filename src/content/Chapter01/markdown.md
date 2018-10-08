
**markdown基本语法**  
<br>
  
  
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

  
## 插入Latex数学公式  


<!-- mathjax config similar to math.stackexchange -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
        inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    messageStyle: "none",
    "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>d

$$
J(\theta) = \frac 1 2 \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$
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
