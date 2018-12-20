
# 人脸生成和处理丢失的标签

我们可以使用GAN（）的有趣应用程序列表是无穷无尽的，我们将在另一个有前景的应用程序中面向生成展示基于CelebA数据库的GANs。我们还将演示如何使用GAN进行半监督学习对标签丢失的数据设置一个标签很差的数据集。                      
本章将介绍以下主题：
- 人脸生成
- 具有生成对抗网络的半监督学习

## 人脸生成

在我们前面提到的章节中，生成器和判别器组成了CNN和DNN。
- CNN是一种神经网络，它将图像的数百个像素编码成一个小尺寸的矢量，这是图像的总结。（集合成一个图像）
- DNN是一个学习一些过滤器以从z恢复原始图像的网络，同样，判别器将输出1或0以指示（说明）输入图像是来自实际数据集还是由生成器生成，另一方面，生成器将尝试复制类似于基于潜在空间z的原始数据集的图像，这可能遵循高斯分布。因此，生成器的目标是学习原始数据集的分布，从而欺骗判别器，从而做出错误的决定。
在这个部分，我们将尝试教生成器学习人脸图片分布，以至于它可以生成真实的人脸。
对于大多数图形公司来说，生成类似人类的面孔是至关重要的，因为它们总是在为自己的应用程序寻找新面孔，在人造面孔方面，它为我们提供了人工智能怎样更加贴近人脸的线索。
在这个例子中，我们将使用CelebA数据集。 CelebFaces属性数据集
（CelebA）是一个大型的脸部属性数据集，有大约200K名人形象，每个都有40个属性注释。数据集涵盖了很多姿势变化，以及背景杂乱，所以CelebA非常多样化，并且有很好的注释。这包括：
- 10,177个身份
- 202,599张脸部图片
- 每个图像有五个地标位置和40个二进制属性注释
我们可以将此数据集用于除面部生成之外的许多计算机视觉应用，例如面部识别和定位，或面部属性检测。此图显示了在训练过程中，生成器发生错误后(或学习人脸分布)是如何接近真实图片的：

![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/01.png?raw=true)


```python
### 获取数据
在本节中，我们将定义一些帮助我们下载的辅助函数CelebA数据集。 我们首先导入他们所需的包,执行命令如下：
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image as pimg
import math
# 有些库会因为之后可能对某个功能进行删减，所以会有警告，而下面这部分的目的就是过滤掉警告
import warnings
warnings.filterwarnings('ignore')
```

### 探索数据
CelebA数据集包含超过20万个带注释的名人图像。 由于我们下面要使用GAN生成类似的图像，因此值得查看数据集中的一堆图像，看看它们的外观。在本节中，我们将定义一些辅助函数，用于可视化CelebA数据集中的一组图像。
现在，让我们使用utils脚本来显示数据集中的一些图像。
该计算机视觉任务的主要目的是使用GAN生成与名人数据集中的图像类似的图像，所以我们需要关注图像的脸部部分。为了关注图像的脸部，我们将删除不包含名人脸部的图像部分。

```python
# 下载数据包celebA的路径
gzip_filename = './data/img_align_celeba.zip'
# 将数据集进行解压缩之后，数据集所在的路径
img_dir = './data/img_align_celeba/'
```

```python
# exits是为了验证数据集的压缩包是否存在，listdir的目的是拿到img_dir目录下所有的文件夹和文件的名字，并将每个都转换成列表中的元素
if os.path.exists(gzip_filename) and len(os.listdir(img_dir)) == 0:
    # 使用zipfile函数中ZipFile方法来打开数据集压缩包，并将此句柄使用gp在之后进行引用
    with zipfile.ZipFile(gzip_filename) as gp:
        # namelist是为了拿到压缩包中所有文件的名字，并针对每个文件进行extract即解压缩处理
        for name in gp.namelist():
            gp.extract(img_dir)
else:
    print('celebA is ready.')
    
    # 展示的数据集中的图片的数量
show_nums = 16
# 建立一个列表，此列表的目的是为了稍后进行存储PIL打开图片的对象
imgs = []
# 计算下总共有多少个图片
count = len(os.listdir(img_dir))
# 设置循环来从硬盘中读取图片
for i in range(show_nums):
    j = i + 1
    # 从硬盘中读取图片，在每次循环都会读取一张，并将对象赋值给img以便之后添加到列表中
    img = pimg.open(img_dir+'{:0>6}.jpg'.format(j))
    # 将读到的图片对象添加到imgs列表中
    imgs.append(img)
# 设置一个4*4的画板，也即是能够显示16个图片的画板
fig,axes = plt.subplots(4,4)
# 对画板进行编号，并对画板中的每个小格显示一张图片
for k,ax in enumerate(axes.flat):
    # 将图片绘画到一个小格中
    ax.imshow(imgs[k-1])
    # 删除刻度
    ax.set_xticks([])
    ax.set_yticks([])
# 展示画板
plt.show()
```
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.1.png?raw=true)
#### 建立模型
现在，让我们从构建实现的核心开始，即开始计算图形。它主要包括以下部分：
- 模型输入
- 判别
- 生成器
- 模型损失
- 模型优化
- 训练模式

```python
# 拿到图片，并将图片进行切割，只显示人的脸部
def get_image(image_path, width, height, mode):

    # 从硬盘中读取图片，并在之后使用image进行引用
    image = pimg.open(image_path)

    # 如果图片的大小不等于只有脸部的样子的话
    if image.size != (width, height):  
        
        # 对脸部的大小进行规划
        face_width = face_height = 108
        # 计算要切割的图片的左侧像素点
        j = (image.size[0] - face_width) // 2
        # 计算要切割的图片的上侧像素点
        i = (image.size[1] - face_height) // 2
        # 对图片进行切割，只保留了图片中脸部
        image = image.crop([j, i, j + face_width, i + face_height])
        # 对图片的边缘进行处理
        image = image.resize([width, height], pimg.BILINEAR)

    # 将裁剪过的图片转换下各个像素点的值，并将每个处理过的图片都用数组的形式作为函数的返回值
    return np.array(image.convert(mode))
```

```python
# 按照批次将图片进行处理
def get_batch(image_files, width, height, mode):
    # 从image_files中依次拿到图片，并将图片转交给get_image函数进行处理，将处理的结果以列表形式储存之后，再转换成数组的形式，其中每个数字都是float32格式
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # 将之前处理过的数据的格式转化成特定的形式（有四个元素的元祖）
    if len(data_batch.shape) < 4:
        a_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch
```

```python
# 设计一个类用来调用之前定义的函数，来处理从硬盘中读取到的图片
class Dataset(object):

    def __init__(self, dataset_name, data_files):

        DATASET_CELEBA_NAME = 'celeba'
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3


        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

    # 按照批次来读取
    def get_batches(self, batch_size):

        IMAGE_MAX_VALUE = 255

        # 用索引的形式进行读取
        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            # 调用之前的函数来读取图片，并将图片进行处理之后返回给data_batch
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            # 更新索引值
            current_index += batch_size

            # 用生成器的形式分批次将处理过的数据迭代出去
            yield data_batch / IMAGE_MAX_VALUE - 0.5
 ```
 
 ```python
 # 将随机生成的数字按照图片的规则进行显示
def images_square_grid(images, mode):


    save_size = math.floor(np.sqrt(images.shape[0]))

    
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # 将图片按照正方形进行排列
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)


    # 使用PIL来重新构建一个图片对象
    new_im = pimg.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            # 从数组中构建一个图片内存
            im = pimg.fromarray(image, mode)
            # 将内存中的图片粘贴到new_im中
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im
 ```
##### 模型输入
在本节中，我们将实现一个辅助函数，这样我们就可以定义模型输入占位符，这个占位符将负责把数据输入计算图。这些辅助函数应该能够创建三个主要占位符：
- 来自数据集的实际输入图像，其中包含的尺寸有（批量大小，输入图像宽度，输入图像高度，通道数）
- 潜在空间Z，将由发生器用于生成假图像学习率占位符
- 辅助函数将返回这三个输入占位符的元组。接下来我们继续定义此功能：

```python
# 定义一个inputs函数
def inputs(img_width,img_height,img_channels,latent_space_z_dim):
    # 使用占位符来定义一个存储样本图片的变量
    true_inputs = tf.placeholder(tf.float32,(None,img_width,img_height,img_channels),'true_inputs')
    # 使用占位符来定义一个存储生成器生成图片的变量
    l_space_inputs = tf.placeholder(tf.float32,(None,latent_space_z_dim),'l_space_inputs')
    # 使用占位符来定义一个模型学习的精度
    model_learning_rate = tf.placeholder(tf.float32,name='model_learning_rate')
    return true_inputs,l_space_inputs,model_learning_rate
```

##### 判别器
接下来，我们需要实现网络的判别器部分，这将用于判断传入的数据是来自真实数据集还是由生成器生成的数据集。同样，我们可以用tf.variable_scope的TensorFlow功能使用判别器为一些变量加前缀，以便于我们可以检索和重新使用它们。所以，让我们定义一个函数，它将返回判别器的二进制输出以及logit值：

```python
# 定义判别器
def discriminator(input_imgs,reuse=False):
    # 创建变量并规定了它的作用域
    with tf.variable_scope('discriminator',reuse=reuse):
        # 设置泄露线性整型激活函数中的参数
        leaky_param_alpha = 0.2
        # 进行第一次卷积操作
        conv_layer_1 = tf.layers.conv2d(input_imgs,64,5,2,'same')
        # 进行第一次激活函数处理
        leaky_relu_output = tf.maximum(leaky_param_alpha*conv_layer_1,conv_layer_1)
        # 进行第二次卷积操作
        conv_layer_2 = tf.layers.conv2d(leaky_relu_output,128,5,2,'same')
        # 用batch_normalization来处理卷积之后的结果，防止模型卡在这个地方
        noramlized_output = tf.layers.batch_normalization(conv_layer_2,training=True)
        # 进行第二次激活函数处理
        leay_relu_output = tf.maximum(leaky_param_alpha*noramlized_output,noramlized_output)
        # 进行第三次卷积神经操作
        conv_layer_3 = tf.layers.conv2d(leay_relu_output,256,5,2,'same')
        # 再次使用batch_normalization处理卷积之后的结果
        noramlized_output = tf.layers.batch_normalization(conv_layer_3,training=True)
        # 进行第三次激活函数处理
        leaky_relu_output = tf.maximum(leaky_param_alpha*noramlized_output,noramlized_output)
        # 将处理过的图片进行压平处理，也即是转换成多维数组
        flattened_output = tf.reshape(leaky_relu_output,(-1,4*4*256))
        # 将压平的数据交给全连接神经网络处理
        logits_layer = tf.layers.dense(flattened_output,1)
        # 将全连接神经网络处理的结果交给sigmoid激活函数进行数据压缩
        output = tf.sigmoid(logits_layer)

        return output,logits_layer
```

##### 生成器
现在，是时候实现网络的第二部分，也就是使用潜在空间z复制原始输入图像。我们也将使用tf.variable_scope来实现此功能。接下来我们定义可以把生成器返回生成图像的函数。

```python
# 定义GAN中的生成器
def generator(z_latent_space,output_channel_dim,is_train=True):
    # 创建变量并规定了它的作用域
    with tf.variable_scope('generator',reuse=not is_train):
        # 设置泄露线性整型激活函数中的参数
        leaky_param_alpha = 0.2
        # 定义全连接层
        fully_connected_layer = tf.layers.dense(z_latent_space,2*2*512)
        # 将全连接层的输出结果进行结构的变化
        reshaped_output = tf.reshape(fully_connected_layer,(-1,2,2,512))
        # 将变化后的数据使用batch_normalization进行处理，防止在训练过程中陷入卡顿
        normalized_output = tf.layers.batch_normalization(reshaped_output,training=is_train)
        # 对上面处理的数据应用激活函数处理
        leaky_relu_output = tf.maximum(leaky_param_alpha*normalized_output,normalized_output)
        # 进行第一次转置卷积
        conv_layer_1 = tf.layers.conv2d_transpose(leaky_relu_output,256,5,2,'same')
        # 对转置卷积之后的结果应用batch_normalization来防止在训练过程中的卡顿
        normalized_output = tf.layers.batch_normalization(conv_layer_1,training=is_train)
        # 对上面进行防卡顿处理之后，再应用泄露线性整型激活函数
        leaky_relu_output = tf.maximum(leaky_param_alpha*normalized_output,normalized_output)
        # 进行第二次转置卷积
        conv_layer_2 = tf.layers.conv2d_transpose(leaky_relu_output,128,5,2,'same')
        # 对第二次转置卷积的结果进行batch_normalization处理
        normalized_output = tf.layers.batch_normalization(conv_layer_2,training=is_train)
        # 对处理过的数据交给激活函数来进行处理
        leaky_relu_output = tf.maximum(leaky_param_alpha*normalized_output,normalized_output)
        # 进行第三次转置卷积
        logits_layer = tf.layers.conv2d_transpose(leaky_relu_output,output_channel_dim,5,2,'same')
        # 将卷积之后的结果交给tanh激活函数进行处理
        output = tf.tanh(logits_layer)

        return output
    
```
##### 模型损失
现在出现了一个棘手的部分，也就是在上一章中讨论过的计算判别器和生成器的损耗。所以，我们定义这样的函数，它将利用先前定义的生成器和判别器函数：

```python
# 定义模型损失
def model_losses(input_actual,input_latent_z,out_channel_dim):
    # 先使用生成器来生成虚假的图片信息
    gen_model = generator(input_latent_z,out_channel_dim)
    # 让判别器进行根据样本信息来进行判断
    disc_model_true,disc_logits_true = discriminator(input_actual)
    # 让判别器对由生成器生成的数据进行判断
    disc_model_fake,disc_logits_fake = discriminator(gen_model,reuse=True)
    # 根据交叉熵来进行模型损失的评估，此处使用的数据是经过判别器判断为真的数据
    disc_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_true,
                                                                            labels=tf.ones_like(disc_model_true)))
    # 根据交叉熵来进行模型损失的评估，此处使用的数据是经过判别器判断为假的数据
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,
                                                                            labels=tf.zeros_like(disc_model_fake)))
    # 根据交叉熵来进行模型损失的评估，此处使用的数据是经过判别器判断为假的数据，但是将标签设置为了真
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,
                                                                      labels=tf.ones_like(disc_model_fake)))
    # 计算总的判别器的损失
    disc_loss = disc_loss_true + disc_loss_fake
    return disc_loss,gen_loss
```
##### 模型优化
最后，在运行我们的模型之前，我们需要实现这个任务的优化标准。下面继续使用之前使用的命名约定来检索判别器和生成器的可训练参数并训练它们：

```python
# 设置模型的优化函数
def model_optimizer(disc_loss,gen_loss,learning_rete,beta11):
    # 将训练过程总使用的变量名全部提取出来
    trainable_vars = tf.trainable_variables()
    # 将训练过程的变量按照命名方式进行分类
    disc_vars = [var for var in trainable_vars if var.name.startswith('discriminator')]
    gen_vars = [var for var in trainable_vars if var.name.startswith('generator')]
    # 根据变量的命名方式，然后对其所代表的训练过程按照要求进行模型优化
    disc_train_opt = tf.train.AdamOptimizer(learning_rete,
                                        beta1=beta11).minimize(disc_loss,var_list=disc_vars)
    # 收集特定的变量名
    update_operations = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 根据变量是否是以generator起始，来进行区分
    gen_updates = [opt for opt in update_operations if opt.name.startswith('generator')]
    # 根据变量之间的相互依赖关系来进行模型的优化
    with tf.control_dependencies(gen_updates):
        gen_train_opt = tf.train.AdamOptimizer(learning_rete,
                                       beta1=beta11).minimize(gen_loss,var_list=gen_vars)

    return disc_train_opt,gen_train_opt
```

```python
# 展示下生成器生成的结果
def show_generator_output(sess,num_images,input_latent_z,output_channel_dim,img_mode):
    # 对图片的色彩数进行设置
    cmap = False if img_mode == 'RGB' else True
    # 将input_latent_z的色彩种类取出来，如果是彩色则取的是3，黑白则是1
    latent_space_z_dim = input_latent_z.get_shape().as_list()[-1]
    # 根据图片的数量和色彩通道数来随机生成新的数组
    examples_z = np.random.uniform(-1,1,size=[num_images,latent_space_z_dim])
    # 根据随机生成的数组，来让生成器处理
    examples = sess.run(generator(input_latent_z,output_channel_dim,False),
                        feed_dict ={input_latent_z:examples_z})
    # 调用images_square_grid将生成器生成的数组进行处理
    images_grid = images_square_grid(examples,img_mode)
    # 将所编辑好的内容加载到画板中
    plt.imshow(images_grid)
    # 展示画板
    plt.show()
```
##### 训练模型
现在，是时候训练模型，看看生成器如何在某种程度上通过生成非常接近原始CelebA数据集的图像来欺骗判别器。
首先，让我们定义一个辅助函数，它将显示一些通过生成器生成的图像：
然后，我们使用之前定义的辅助函数来构建模型输入，损失和优化标准。我们将它们堆叠在一起，并开始基于CelebA数据集训练我们的模型。启动培训的过程，可能需要一些时间，它取决于每个人的主机规格。
经过一段时间对模型的训练，我们可以得到下面的图形。
```python
# 定义模型训练函数
def model_train(epochs,batch_size,learning_rete,beta11,get_batches,input_data_shape,data_img_mode):
    # 将数据集的格式交给单独的变量
    image_width,image_height,image_channels,z_dims = input_data_shape
    # 调用inputs函数，并将返回的结果交给相对应的变量
    actual_input,z_input,learningRate = inputs(image_width,image_height,image_channels,z_dims)
    # 调用model_losses函数，并将返回的结果交给相应的变量
    disc_loss,gen_loss = model_losses(actual_input,z_input,image_channels)
    # 调用model_optimizer函数，并将返回的结果交给相应的变量
    disc_opt,gen_opt = model_optimizer(disc_loss,gen_loss,learning_rete,beta11)
    # 循环次数变量
    steps = 0
    # 每循环50次，打印一次模型的损失
    print_every = 50
    # 每循环100次，打印一下生成器生成的图片
    show_every = 100
    # 设置每次循环时产生的模型损失
    model_loss = []

    with tf.Session() as sess:
        # 对所有的tensorflow的变量进行初始化，准备开始进行训练
        sess.run(tf.global_variables_initializer())
        # 根据训练周期进行循环
        for epoch in range(epochs):
            # 根据数据集的批次来进行循环
            for batch_images in get_batches(batch_size):
                # 计算循环的次数
                steps += 1
                batch_images *= 2.0
                # 生成随机数据
                z_sample = np.random.uniform(-1,1,(batch_size,z_dims))
                # 对模型进行优化
                _ = sess.run(disc_opt,feed_dict={actual_input:batch_images,z_input:z_sample,learningRate:learning_rete})
                _ = sess.run(gen_opt,feed_dict={z_input:z_sample,learningRate:learning_rete})
                # 每训练50次计算一次模型损失，并显示出来
                if steps % print_every == 0:
                    train_loss_disc = disc_loss.eval({z_input:z_sample,actual_input:batch_images})
                    train_loss_gen = gen_loss.eval({z_input:z_sample})

                    print('Epoch {}/{}\t'.format(epoch + 1,epochs),
                          'Discriminator Loss:{:.4f}\t'.format(train_loss_disc),
                          'Generator Loss:{:.4f}'.format(train_loss_gen))

                    model_loss.append((train_loss_disc,train_loss_gen))
                    # 每训练100次展示一次生成器生成的图片
                if steps % show_every == 0:
                    show_generator_output(sess,num_images,z_input,image_channels,data_img_mode)
```

```python
# 每次训练从硬盘中读取的数据量
train_batch_size = 512
# 生成器生成的数组，与图片的大小有关
z_dim = 100
# 在进行模型优化时的模型精确度
learning_rate = 0.002
# 模型优化时的一个参数
beta1 = 0.5
# 只进行两个周期的训练
num_epochs = 2
```

```python
# 对数据集进行实例化
celeba_dataset = Dataset('celeba',glob(os.path.join(img_dir,'*.jpg')))
# 设置一个上下文管理器
with tf.Graph().as_default():
    # 调用model_train开始进行训练
    model_train(num_epochs, train_batch_size,    learning_rate,  beta1,
                celeba_dataset.get_batches, [28,28,3,train_batch_size],   'RGB')
```

 Epoch 1/2	 Discriminator Loss:0.0032	 Generator Loss:7.0822
    Epoch 1/2	 Discriminator Loss:0.0012	 Generator Loss:9.5755
    ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.2.png?raw=true)
    
     Epoch 1/2	 Discriminator Loss:0.0010	 Generator Loss:9.1579
    Epoch 1/2	 Discriminator Loss:0.0006	 Generator Loss:9.5570
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.3.png?raw=true)
    
     Epoch 1/2	 Discriminator Loss:0.0005	 Generator Loss:10.2887
    Epoch 1/2	 Discriminator Loss:0.0009	 Generator Loss:10.5652
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.4.png?raw=true)
    
     Epoch 1/2	 Discriminator Loss:0.0003	 Generator Loss:10.9225
    Epoch 2/2	 Discriminator Loss:0.0002	 Generator Loss:11.8593
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.5.png?raw=true)
   
     Epoch 2/2	 Discriminator Loss:0.0001	 Generator Loss:12.7685
    Epoch 2/2	 Discriminator Loss:0.0001	 Generator Loss:12.2158
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.6.png?raw=true)
    
     Epoch 2/2	 Discriminator Loss:0.0001	 Generator Loss:12.7092
    Epoch 2/2	 Discriminator Loss:0.0001	 Generator Loss:13.2471
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.7.png?raw=true)
    
     Epoch 2/2	 Discriminator Loss:0.0000	 Generator Loss:13.4493
    Epoch 2/2	 Discriminator Loss:0.0000	 Generator Loss:13.7767
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/1.8.png?raw=true)
   
     Epoch 2/2	 Discriminator Loss:0.0000	 Generator Loss:13.5980
    
    
    
    


# 生成对抗网络的半监督学习

半监督学习是一种技术，其中标记和未标记的数据都用来训练分类器。
这种类型的分类器占用标记数据的一小部分和大量未标记数据（来自同一域）。它的目的是将这些数据源结合起来训练深度卷积神经网络（DCNN），来学习能够将新数据点映射到其期望结果的推断函数。
在这个前沿，我们提出了一个GAN模型，使用一个非常小的标记训练集对街景房屋号码进行分类。实际上，该模型使用大约1.3％的原始SVHN训练标签，也就是1000（一千）标记的示例。我们使用一些从OpenAI（网站中找到）（http://arxiv.org/abs/1606.03498）

### 直觉
在构建用于生成图像的GAN时，我们同时训练了生成器和判别器。训练后，我们可以丢弃判别器，因为我们只是用它来训练生成器。
下图是使用半监督学习GAN对11种分类问题的体系结构。
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/02.png?raw=true)
在半监督学习中，我们需要将判别器转换为多类分类器。这个新模型必须能够在测试集上很好地概括，即使我们没有很多用于训练的标记示例。另外，这次，在训练结束时，我们实际上可以扔掉生成器。这时候的角色已经更改。现在生成器仅用于在训练期间帮助判别器。换句话说，生成器充当不同的信息源，判别器从中获得原始的，未标记的训练数据。正如我们将看到的，这些未标记的数据是提高判别器性能的关键。此外，对于常规图像生成GAN，判别器仅具有一个角色。计算其输入是真实的还是非真实的,我们把它称为GAN问题的概率。
然而，为了将判别器变为半监督分类器，除了GAN问题之外，判别器还必须学习每个原始数据集类的概率。换句话说，对于每个输入的图像，判别器必须学习它的概率。
回想一下，对于图像生成GAN判别器，我们有一个sigmoid单位输出。这个输出值表示输入图像为真实的概率（值接近1），或假的概率（值接近0）。换句话说，从判别器的角度来看，接近1的值意味着样本可能来自训练集。同样，接近0的值意味着样本来自生成器网络的可能性更高。通过使用该概率，判别器能够将信号发送回生成器。这个信号允许生成器在训练期间调整其参数，从而可以提高其创建逼真图像的能力。
我们必须将判别器（从之前的GAN）转换为11类分类器。为此，我们可以将其sigmoid输出转换为具有11级输出的softmax，前10位为SVHN数据集的个体类概率（零到九），以及来自生成器的所有伪图像的第11类。
注意：如果我们将第11类概率设置为0，则前10个概率的总和表示使用sigmoid函数计算的相同概率。
最后，我们需要设置损失，以便判别器可以同时做到：
- 帮助生成器学习生成逼真的图像。 为此，我们必须指示判别器区分真假样本。
- 使用生成器生成的图像以及标记和未标记的训练数据来帮助对数据集进行分类。

总而言之，判别器有三种不同的训练数据来源：
- 带标签的真实图像。（这些是像任何常规监督分类问题中的图像标签对。
- 没有标签的真实图像。（对于那些，判别器只学习这些图像是真实的。）
- 来自生成器的图像。（为了使用这些，判别者学会将其归类为假的。）
这些不同数据源的组合将使分类器能够从更广泛的角度进行学习。 反过来，这使得模型能够比仅使用1,000个标记的示例进行训练时更精确地执行推理。
数据分析和预处理

```python
import pickle as pkl
import time
# 画图工具
import matplotlib.pyplot as plt
import numpy as np
# 加载mat格式文件的库
from scipy.io import loadmat
# 深度学习的架构
import tensorflow as tf

extra_class = 0

# 下载数据集所需要的库
from urllib.request import urlretrieve
# 判断数据集是否存在的库
from os.path import isfile, isdir
# 显示进度条的库
from tqdm import tqdm


input_data_dir = 'input/'
```

```python
# 查看存放数据的文件夹是否存在，不存在就抛出异常
if not isdir(input_data_dir):
    raise Exception("Data directory doesn't exist!")

# 定义一个新类，让这个新类继承tqdm这个类，在这里目的是为了下载数据时可以显示进度条
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

# 查看train_32x32.mat是否存在，如果不存在，就从其它地方把它下载下来
if not isfile(input_data_dir + "train_32x32.mat"):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
        urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            input_data_dir + 'train_32x32.mat',
            pbar.hook)

# 查看test_32x32.mat是否存在，如果不存在，就从其它地方把它下载下来
if not isfile(input_data_dir + "test_32x32.mat"):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
        urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            input_data_dir + 'test_32x32.mat',
            pbar.hook)
```

### 数据分析和预处理
在这个任务中，我们将使用SVHN数据集，它是斯坦福的街景房屋编号的缩写。因此，让我们开始该实现通过导入所需要的包开始：
接下来，我们将定义一个帮助类来下载SVHN数据集（首先需要手动创建JOQVU@EBUB@EJS first）：
让我们了解这些图像是什么样子的。

```python
# 加载用来训练的数据集
train_data = loadmat(input_data_dir + 'train_32x32.mat')
# 加载用来测试的测试集
test_data = loadmat(input_data_dir + 'test_32x32.mat')

# 生成随机的整型数字
indices = np.random.randint(0, train_data['X'].shape[3], size=36)
# 设置6*6的画板
fig, axes = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(5,5),)
for ii, ax in zip(indices, axes.flatten()):
    ax.imshow(train_data['X'][:,:,:,ii], aspect='equal')
    # 不显示图片的轴
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
# 自动调整大小并展示画板结果
plt.subplots_adjust(wspace=0, hspace=0)
```

 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.1.png?raw=true)
接下来，我们需要将图像缩到-1到1之间，并且这个步骤是必要的，因此我们将使用tanh()函数，它将压缩生成器的输出值：

```python
# 将输入的图片进行缩放
def scale_images(image, feature_range=(-1, 1)):
    # 将图片上的每个像素点的值都缩放到0-1之间
    image = ((image - image.min()) / (255 - image.min()))

    # 将图片的值缩放到特征范围
    min, max = feature_range
    image = image * (max - min) + min
    return image
```

#### 建立模型 
在本节中，我们将构建测试所需要的所有部分，因此我们从定义输入开始，这些输入将用于给计算图提供数据。

```python
class Dataset:
    def __init__(self, train_set, test_set, validation_frac=0.5, shuffle_data=True, scale_func=None):
        
        split_ind = int(len(test_set['y']) * (1 - validation_frac))
        
        self.test_input, self.valid_input = test_set['X'][:, :, :, :split_ind], test_set['X'][:, :, :, split_ind:]
        
        self.test_target, self.valid_target = test_set['y'][:split_ind], test_set['y'][split_ind:]
        
        self.train_input, self.train_target = train_set['X'], train_set['y']

 
        # 因为门牌号都是由很多标签，但是我们在这里只假设有1000个
        self.label_mask = np.zeros_like(self.train_target)
        
        self.label_mask[0:1000] = 1

        self.train_input = np.rollaxis(self.train_input, 3)
        
        self.valid_input = np.rollaxis(self.valid_input, 3)
        
        self.test_input = np.rollaxis(self.test_input, 3)

        if scale_func is None:
            self.scaler = scale_images
        else:
            self.scaler = scale_func
            
        self.train_input = self.scaler(self.train_input)
        
        self.valid_input = self.scaler(self.valid_input)
        
        self.test_input = self.scaler(self.test_input)
        
        self.shuffle = shuffle_data

    # 加载数据并处理 
    def batches(self, batch_size, which_set="train"):
        input_name = which_set + "_input"
        target_name = which_set + "_target"

        # 获得target_name所代表的属性值的长度
        num_samples = len(getattr(dataset, target_name))
        if self.shuffle:
            # 对长度进行索引
            indices = np.arange(num_samples)
            # 将索引进行打乱顺序
            np.random.shuffle(indices)
            # 重新设置input_name和target_name所对应的属性值
            setattr(dataset, input_name, getattr(dataset, input_name)[indices])
            setattr(dataset, target_name, getattr(dataset, target_name)[indices])
            if which_set == "train":
                dataset.label_mask = dataset.label_mask[indices]

        # 提取dataset中对于变量的属性值
        dataset_input = getattr(dataset, input_name)
        dataset_target = getattr(dataset, target_name)

        # 设置循环，依批次取出数据集中的数据
        for jj in range(0, num_samples, batch_size):
            # 根据索引来取出数据集中的数据
            input_vals = dataset_input[jj:jj + batch_size]
            target_vals = dataset_target[jj:jj + batch_size]

            if which_set == "train":
                # 包含了label_mask，以防止训练集中我们不知道的标签
                # 通过生成器来批次将获得的数据提取出来
                yield input_vals, target_vals, self.label_mask[jj:jj + batch_size]
            else:
                yield input_vals, target_vals
```

##### 模型输入
首先，我们将定义模型输入参数，它将用于给计算模型提供数据的模型输入占位符

```python
# 定义模型中的输入变量
def inputs(actual_dim, z_dim):
    # 使用占位符来定义之后需要的变量
    inputs_actual = tf.placeholder(tf.float32, (None, *actual_dim), name='input_actual')
    inputs_latent_z = tf.placeholder(tf.float32, (None, z_dim), name='input_latent_z')

    target = tf.placeholder(tf.int32, (None), name='target')
    label_mask = tf.placeholder(tf.int32, (None), name='label_mask')

    return inputs_actual, inputs_latent_z, target, label_mask
```

##### 生成器
在这一部分中，我们将实现GAN网络的第一个核心部分。本部分的结构和实现将遵循最初的DCGAN文件：

```python
# 定义生成器
def generator(latent_z, output_image_dim, reuse_vars=False, leaky_alpha=0.2, is_training=True, size_mult=128):
    with tf.variable_scope('generator', reuse=reuse_vars):
        # 定义一个全连接的神经网络
        fully_conntected_1 = tf.layers.dense(latent_z, 4 * 4 * size_mult * 4)

        # 把全连接神经网络的结果由两维转换成四维
        reshaped_out_1 = tf.reshape(fully_conntected_1, (-1, 4, 4, size_mult * 4))
        # 使用batch_normalization来对数据进行处理，防止在训练过程中卡顿
        batch_normalization_1 = tf.layers.batch_normalization(reshaped_out_1, training=is_training)
        # 使用泄露线性整型激活函数来处理上述数据
        leaky_output_1 = tf.maximum(leaky_alpha * batch_normalization_1, batch_normalization_1)
        # 进行转置卷积操作
        conv_layer_1 = tf.layers.conv2d_transpose(leaky_output_1, size_mult * 2, 5, strides=2, padding='same')
        # 使用batch_normalization来对数据进行处理
        batch_normalization_2 = tf.layers.batch_normalization(conv_layer_1, training=is_training)
        # 激活函数进行处理
        leaky_output_2 = tf.maximum(leaky_alpha * batch_normalization_2, batch_normalization_2)
        # 第二次转置卷积操作
        conv_layer_2 = tf.layers.conv2d_transpose(leaky_output_2, size_mult, 5, strides=2, padding='same')
        # 使用batch_normalization来对数据进行处理
        batch_normalization_3 = tf.layers.batch_normalization(conv_layer_2, training=is_training)
        # 激活函数进行处理
        leaky_output_3 = tf.maximum(leaky_alpha * batch_normalization_3, batch_normalization_3)

        # 定义输出层
        logits_layer = tf.layers.conv2d_transpose(leaky_output_3, output_image_dim, 5, strides=2, padding='same')
        # 将输出层交给tanh激活函数处理
        output = tf.tanh(logits_layer)

        return output
```
##### 判别器
现在，是时候建立GAN网络的第二个核心部分了——判别器。在以前的实现中，我们说过判别器将产生一个二进制输出，表示输入的图像来自真实数据集（1）还是由生成器（0）生成的数据集。这里的情况不同，所以判别器现在将是一个多类分类器。
现在，让我们继续构建体系结构的判别器部分：
不是在最后应用一个完全连接的层，我们将执行所谓的全球平均池（GAP），GAP取一个特征向量的空间维数的平均值。这将产生一个压缩的张量到一个值。
例如，假设在一些卷积之后，我们得到了一个形状的输出张量：

> [BATCH_SIZE,8,8,NUM_CHANNELS]

为了应用全局变量，我们计算了8*8张量切片上的平均值。此操作将产生一个张量，其形状如下

> [BATCH_SIZE,1,1,NUM_CHANNELS]

在应用全局平均池之后，我们添加一个输出最终逻辑的完全连接的层，这些层形如下所示

>[BATCH_SIZE,NUM_CLASSES]

这些层代表每一个分类的得分。为了得到概率的分数，我们将使用TPGUNBY激活函数：
最后，判别器函数将如下所示，

```python
# 定义网络中的判别器
def discriminator(input_x, reuse_vars=False, leaky_alpha=0.2, drop_out_rate=0., num_classes=10, size_mult=64):
    # 定义变量的作用域
    with tf.variable_scope('discriminator', reuse=reuse_vars):
        # 定义为了防止过度拟合而随机丢弃的数据比例
        drop_out_output = tf.layers.dropout(input_x, rate=drop_out_rate / 2.5)

        # 对输入的数据进行卷积操作、激活函数处理、随机丢弃一部分数据等操作
        conv_layer_3 = tf.layers.conv2d(input_x, size_mult, 3, strides=2, padding='same')
        leaky_output_4 = tf.maximum(leaky_alpha * conv_layer_3, conv_layer_3)
        leaky_output_4 = tf.layers.dropout(leaky_output_4, rate=drop_out_rate)

        conv_layer_4 = tf.layers.conv2d(leaky_output_4, size_mult, 3, strides=2, padding='same')
        batch_normalization_4 = tf.layers.batch_normalization(conv_layer_4, training=True)
        leaky_output_5 = tf.maximum(leaky_alpha * batch_normalization_4, batch_normalization_4)

        conv_layer_5 = tf.layers.conv2d(leaky_output_5, size_mult, 3, strides=2, padding='same')
        batch_normalization_5 = tf.layers.batch_normalization(conv_layer_5, training=True)
        leaky_output_6 = tf.maximum(leaky_alpha * batch_normalization_5, batch_normalization_5)
        
        leaky_output_6 = tf.layers.dropout(leaky_output_6, rate=drop_out_rate)

        conv_layer_6 = tf.layers.conv2d(leaky_output_6, 2 * size_mult, 3, strides=1, padding='same')
        batch_normalization_6 = tf.layers.batch_normalization(conv_layer_6, training=True)
        leaky_output_7 = tf.maximum(leaky_alpha * batch_normalization_6, batch_normalization_6)

        conv_layer_7 = tf.layers.conv2d(leaky_output_7, 2 * size_mult, 3, strides=1, padding='same')
        batch_normalization_7 = tf.layers.batch_normalization(conv_layer_7, training=True)
        leaky_output_8 = tf.maximum(leaky_alpha * batch_normalization_7, batch_normalization_7)

        conv_layer_8 = tf.layers.conv2d(leaky_output_8, 2 * size_mult, 3, strides=2, padding='same')
        batch_normalization_8 = tf.layers.batch_normalization(conv_layer_8, training = True)
        leaky_output_9 = tf.maximum(leaky_alpha * batch_normalization_8, batch_normalization_8)
        
        leaky_output_9 = tf.layers.dropout(leaky_output_9, rate = drop_out_rate)

        conv_layer_9 = tf.layers.conv2d(leaky_output_9, 2 * size_mult, 3, strides=1, padding='same')
        leaky_output_10 = tf.maximum(leaky_alpha * conv_layer_9, conv_layer_9)

        # 对上述处理的结果，计算各个维度的平均值
        leaky_output_features = tf.reduce_mean(leaky_output_10, (1, 2))

        # 设置一个变量来接收神经网络的结果
        classes_logits = tf.layers.dense(leaky_output_features, num_classes + extra_class)

        if extra_class:
            actual_class_logits, fake_class_logits = tf.split(classes_logits, [num_classes, 1], 1)
            assert fake_class_logits.get_shape()[1] == 1, fake_class_logits.get_shape()
            fake_class_logits = tf.squeeze(fake_class_logits)
        else:
            actual_class_logits = classes_logits
            fake_class_logits = 0.

        # 计算各个维度上的最大值
        max_reduced = tf.reduce_max(actual_class_logits, 1, keep_dims=True)
        
        stable_actual_class_logits = actual_class_logits - max_reduced

        gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_actual_class_logits), 1)) + tf.squeeze(
            max_reduced) - fake_class_logits

        # 对上述的结果进行softmax处理
        softmax_output = tf.nn.softmax(classes_logits)

        return softmax_output, classes_logits, gan_logits, leaky_output_features
```

##### 模型损失
现在是定义模型损失的时候了。首先，判别器损失将分为两部分： 
- 第一种代表GAN问题，即无监督损失。 
- 第二种是计算个体的实际类概率，即监督损失。对于判别器的无监督损失，需要对实际的训练图像和生成图像进行区分。
对于一般的GAN，一半时间内，判别器会从训练集中获取未标记的图像作为输入，另一半则从生成器获取假的、未标记的图像。
对于判别器损失的第二部分，即监督损失，它需要建立在判别器逻辑的基础上。因此，我们将使用softmax交叉熵，因为这是一个多分类问题。
最后，如下所示：

```python
# 定义模型损失
def model_losses(input_actual, input_latent_z, output_dim, target, num_classes, label_mask, leaky_alpha=0.2,
                     drop_out_rate=0.):

        # 定义了生成器和判别器的矩阵大小
        gen_size_mult = 32
        disc_size_mult = 64

        # 运行生成器和判别器
        gen_model = generator(input_latent_z, output_dim, leaky_alpha=leaky_alpha, size_mult=gen_size_mult)
        disc_on_data = discriminator(input_actual, leaky_alpha=leaky_alpha, drop_out_rate = drop_out_rate,size_mult = disc_size_mult)
        
        
        disc_model_real, class_logits_on_data, gan_logits_on_data, data_features = disc_on_data
        # 在生成器生成的结果上运行判别器
        disc_on_samples = discriminator(gen_model, reuse_vars=True, leaky_alpha=leaky_alpha,
                                        drop_out_rate=drop_out_rate, size_mult=disc_size_mult)
        
        disc_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = disc_on_samples

        # 使用交叉熵，计算了样本中被判为真的损失
        disc_loss_actual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                    labels=tf.ones_like(gan_logits_on_data)))
        # 使用交叉熵，计算了样本中被判为假的损失
        disc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                    labels=tf.zeros_like(gan_logits_on_samples)))
        # 将target变量中各个维度中只有1个元素的维度清除掉
        target = tf.squeeze(target)
        classes_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                                        labels=tf.one_hot(target,num_classes + extra_class,dtype=tf.float32))
        
        classes_cross_entropy = tf.squeeze(classes_cross_entropy)
        
        label_m = tf.squeeze(tf.to_float(label_mask))
        
        disc_loss_class = tf.reduce_sum(label_m * classes_cross_entropy) / tf.maximum(1., tf.reduce_sum(label_m))
        
        disc_loss = disc_loss_class + disc_loss_actual + disc_loss_fake

        # 设置了sample_features、data_features中各个维度的平均值
        sampleMoments = tf.reduce_mean(sample_features, axis=0)
        dataMoments = tf.reduce_mean(data_features, axis=0)

        gen_loss = tf.reduce_mean(tf.abs(dataMoments - sampleMoments))
        # cast将后面括号中的第一个元素中各个元素转换成第二个元素所代表的格式
        prediction_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
        # 对括号中两个数组进行对比，看是否相等
        check_prediction = tf.equal(tf.squeeze(target), prediction_class)
        # 对各个维度进行求和
        correct = tf.reduce_sum(tf.to_float(check_prediction))
        # 对各个维度进行求和
        masked_correct = tf.reduce_sum(label_m * tf.to_float(check_prediction))

        return disc_loss, gen_loss, correct, masked_correct, gen_model
```

```python
def model_optimizer(disc_loss, gen_loss, learning_rate, beta1):

        # 获得所有的变量并将它们以名字的开头的格式进行区分
        trainable_vars = tf.trainable_variables()
        disc_vars = [var for var in trainable_vars if var.name.startswith('discriminator')]
        gen_vars = [var for var in trainable_vars if var.name.startswith('generator')]
        
        for t in trainable_vars:
            assert t in disc_vars or t in gen_vars

        # 优化生成器和判别器的结果的损失
        disc_train_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(disc_loss,var_list=disc_vars)
        gen_train_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(gen_loss, var_list=gen_vars)
        # 将learning_rate更新为原来的0.9倍
        shrink_learning_rate = tf.assign(learning_rate, learning_rate * 0.9)

        return disc_train_optimizer, gen_train_optimizer, shrink_learning_rate
```

##### 模型优化器
现在，我们定义模型优化器，它与我们前面定义的模型非常相似

```python
# 定义生成对抗性网络
class GAN:
        def __init__(self, real_size, z_size, learning_rate, num_classes=10, alpha=0.2, beta1=0.5):
            
            tf.reset_default_graph()

            self.learning_rate = tf.Variable(learning_rate, trainable=False)
            
            model_inputs = inputs(real_size, z_size)
            
            self.input_actual, self.input_latent_z, self.target, self.label_mask = model_inputs
            
            self.drop_out_rate = tf.placeholder_with_default(.5, (), "drop_out_rate")
            # 计算模型损失
            losses_results = model_losses(self.input_actual, self.input_latent_z,
                                          real_size[2], self.target, num_classes,
                                          label_mask=self.label_mask,
                                          leaky_alpha=0.2,
                                          drop_out_rate= self.drop_out_rate)
            
            self.disc_loss, self.gen_loss, self.correct, self.masked_correct, self.samples = losses_results
            # 对模型进行优化
            self.disc_opt, self.gen_opt, self.shrink_learning_rate = model_optimizer(self.disc_loss, self.gen_loss,
                                                                                     self.learning_rate, beta1)

```

```python
# 定义函数来显示图片
def view_generated_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):
    
        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,sharey=True, sharex=True)
        
        for ax, img in zip(axes.flatten(), samples[epoch]):
            ax.axis('off')
            img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
            ax.set_adjustable('box-forced')
            im = ax.imshow(img)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        return fig, axes
```

```python
# 定义训练函数
def train(net, dataset, epochs, batch_size, figsize=(5, 5)):
    
        saver = tf.train.Saver()
        sample_z = np.random.normal(0, 1, size=(50, latent_space_z_size))

        samples, train_accuracies, test_accuracies = [], [], []
        steps = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 根据周期进行训练
            for e in range(epochs):
                print("Epoch", e )

                num_samples = 0
                num_correct_samples = 0
                # 根据批次进行训练
                for x, y, label_mask in dataset.batches(batch_size):
                    assert 'int' in str(y.dtype)
                    steps += 1
                    num_samples += label_mask.sum()

                    # 根据要求随机生成一些数组
                    batch_z = np.random.normal(0, 1, size=(batch_size, latent_space_z_size))

                    _, _, correct = sess.run([net.disc_opt, net.gen_opt, net.masked_correct],
                                             feed_dict={net.input_actual: x, net.input_latent_z: batch_z,
                                                        net.target: y, net.label_mask: label_mask})
                    
                    num_correct_samples += correct

                sess.run([net.shrink_learning_rate])
                # 计算在训练集中的准确率
                training_accuracy = num_correct_samples / float(num_samples)

                print("\t\tClassifier train accuracy: ", training_accuracy)

                num_samples = 0
                num_correct_samples = 0
                # 从测试集中依批次拿到数据并进行测试准确率
                for x, y in dataset.batches(batch_size, which_set="test"):
                    assert 'int' in str(y.dtype)
                    num_samples += x.shape[0]

                    correct, = sess.run([net.correct], feed_dict={net.input_actual: x,
                                                                  net.target: y,
                                                                  net.drop_out_rate: 0.})
                    num_correct_samples += correct
                # 计算在测试集上的准确率
                testing_accuracy = num_correct_samples / float(num_samples)
                print("\t\tClassifier test accuracy", testing_accuracy)

                gen_samples = sess.run(net.samples,feed_dict={net.input_latent_z: sample_z})
                
                samples.append(gen_samples)
                
                _,_ = view_generated_samples(-1, samples, 5, 10, figsize=figsize)


                # 将准确率保存下来，接下来可能要看
                train_accuracies.append(training_accuracy)
                test_accuracies.append(testing_accuracy)
            # 将会话给保存下来
            saver.save(sess, './checkpoints/generator.ckpt')

        # 将samples的属性值保存到文件samples.pkl中
        with open('samples.pkl', 'wb') as f:
            pkl.dump(samples, f)
        return train_accuracies, test_accuracies, samples
```

```python
# 定义数据集的大小
real_size = (32,32,3)
# 定义生成器生成的长度
latent_space_z_size = 100
# 学习的准确度
learning_rate = 0.0003
# 对GAN进行实例化
net = GAN(real_size, latent_space_z_size, learning_rate)
# 对Dataset进行实例化
dataset = Dataset(train_data, test_data)
# 每次训练时拿的数据量
train_batch_size = 128
# 总共进行5个周期的训练
num_epochs = 5

# 开始训练
train_accuracies, test_accuracies, samples = train(net,dataset,num_epochs,train_batch_size,figsize=(10,5))

# 设置画布，并显示模型在训练集和测试集上的准确率
fig, ax = plt.subplots()
plt.plot(train_accuracies, label='Train', alpha=0.5)
plt.plot(test_accuracies, label='Test', alpha=0.5)
plt.title("Accuracy")
plt.legend()
```

WARNING:tensorflow:From <ipython-input-8-6588728a804a>:55: calling reduce_max (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From <ipython-input-9-1cee69543cc5>:32: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

Epoch 0
		Classifier train accuracy:  0.179
		Classifier test accuracy 0.24746465888137675
    
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.2.png?raw=true)

    Epoch 1
		Classifier train accuracy:  0.339
		Classifier test accuracy 0.433159188690842
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.3.png?raw=true)
    
   Epoch 2
		Classifier train accuracy:  0.539
		Classifier test accuracy 0.5295021511985248
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.4.png?raw=true)
    
    Epoch 3
		Classifier train accuracy:  0.681
		Classifier test accuracy 0.6194683466502766
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.5.png?raw=true)
    
    Epoch 4
		Classifier train accuracy:  0.807
		Classifier test accuracy 0.6441303011677935
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.6.png?raw=true)
    
     <matplotlib.legend.Legend at 0x217ca6e2358>
  
   ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.7.png?raw=true)  
    
    ```python
    # 此处会将之前显示的图片直接保存到电脑的硬盘中
    for ii in range(len(samples)):
    fig,ax = view_generated_samples(ii, samples, 5, 10, figsize=(10,5))
    fig.savefig('images/samples_{:03d}.png'.format(ii))
    plt.close()
   ```
    
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.8.png?raw=true)
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.9.png?raw=true)
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.10.png?raw=true)
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.11.png?raw=true)
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter15/chapter15_image/2.12.png?raw=true)
 
   虽然特征匹配损失在半监督学习的任务中表现良好，但是生成器生成的图像不如前面章节创建的图像好。但是这个实现主要是为了演示我们如何使用用于半监督学习设置的GAN。

## 总结
最后，许多研究人员认为无监督学习是一般AI中缺失的环节系统。为克服这些障碍，尝试用较少的方法解决已确定的问题，标记数据是关键。在这种情况下，GAN为复杂学习提供了一个真正的选择，标记较少的样本的任务。然而，监督和半监督之间的绩效差距学习仍然远非平等。 我们当然可以期待这种差距随着新方法的发挥而变得更小。

学号|姓名|专业
-|-|-
201802110533|张倩茹|概率论与数理统计
201802110535|王悦|概率论与数理统计
<br>

    
    
