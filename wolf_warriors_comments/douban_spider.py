# -*- coding: utf-8 -*-
import requests 
import re
import pandas as pd

url_first = 'https://movie.douban.com/subject/26363254/comments?start=0'  # start page
# head = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0'}
head = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36'}
cookies = {'cookie':'bid=gnZm7kpcBEU; __utmt=1; ps=y; ue="2680819187@qq.com"; dbcl2="166508389:/zihTbwT1xs"; ck=rYrw; __utma=30149280.700482388.1504868551.1504868551.1504868551.1; __utmb=30149280.1.10.1504868551; __utmc=30149280; __utmz=30149280.1504868551.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utma=223695111.42541366.1504868560.1504868560.1504868560.1; __utmb=223695111.0.10.1504868560; __utmc=223695111; __utmz=223695111.1504868560.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); _pk_id.100001.4cf6=b81e986601aab223.1504868550.1.1504868583.1504868550.; _pk_ses.100001.4cf6=*; push_noty_num=0; push_doumail_num=0'}  #cookie of your account

html = requests.get(url_first, headers=head, cookies=cookies)  # get first page

re_page = re.compile(r'<a href="(.*?)&amp;.*?class="next">') # next page

re_content = re.compile(r'<span class="votes">(.*?)</span>.*?comment">(.*?)</a>.*?</span>.*?<span.*?class="">(.*?)</a>.*?<span>(.*?)</span>.*?title="(.*?)"></span>.*?title="(.*?)">.*?class=""> (.*?)\n', re.S)

while html.status_code==200:
    url_next = 'https://movie.douban.com/subject/26363254/comments' + re.findall(re_page, html.text)[0]
    data = re.findall(re_content, html.text)
    print(url_next)
    print(data)
    frame = pd.DataFrame(data)
    # frame.to_csv('/home/zhanghao/Desktop/comments.csv', header=False, index=False, mode='a+', encoding='utf-8')
    frame.to_csv('./data/comments.csv', header=False, index=False, mode='a+', encoding='utf-8')
    frame = []
    data = []
    html = requests.get(url_next, cookies=cookies, headers=head)