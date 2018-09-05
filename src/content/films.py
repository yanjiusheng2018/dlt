#-*-coding:utf8-*-
import re
import requests
import sys

#编码转换
reload(sys)
sys.setdefaultencoding('utf-8')
url='http://maoyan.com/films?showType=1&offset=30'
with open('D:/xxx/xxx.txt', 'a') as f:
#将多数页码的源代码合并写入自动创建的文件，让目标网站以为是浏览器，并非爬虫
   for page in range(0,271,30):
      new_url=re.sub('offset=\d+','offset=%d'%page,url,re.S)
      head={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36'}
      html=requests.get(new_url,headers=head)
      html.encoding='utf-8'
      f.write(html.text)
#遍历源代码中的图片链接
ff = open('D:/xxx/xxx.txt','r')
Html = ff.read()
ff.close()
pic_url = re.findall('<img data-src="(.*?)" />',Html,re.S)
i = 0
for each in pic_url:
    print 'now downloading:' + each
    pic = requests.get(each)
    fp = open('D:/xxx/films\\' + str(i) + '.jpg','wb')
    fp.write(pic.content)
    fp.close()
    i += 1
