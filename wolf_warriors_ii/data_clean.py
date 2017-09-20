import re
import pandas as pd

reg_correct = re.compile(r'(.*),(.*),(.*),(.*),(.*),(.*),(.*)')
reg_dirty = re.compile(r'.*这条短评跟影片无关.*<span class=""votes"">(.*?)</span>.*?class=""j a_vote_comment"">(.*?)</a>.*?class="""">(.*?)</a><span>(.*?)</span>.*?title=""(.*?)",(.*),(.*)')
data = []  # 格式化数据

# 第一步，清理错误格式数据，从错误格式数据中提取正确数据，并加入到dataframe中
with open('./data/comments.csv') as file:
    dirty_data = []  # 错误格式数据
    dirty_lines = ''
    flag = 0
    # load and clean data
    for line in file:
        arr = re.findall(reg_correct, line)
        if len(arr) == 0:
            dirty_lines = ''.join([dirty_lines, line.strip()])
            flag = 1
        else:
            value = (arr[0][0].replace('"', ''), arr[0][1].replace('"', ''), arr[0][2].replace('"', ''), arr[0][3].replace('"', ''), arr[0][4].replace('"', ''), arr[0][5].replace('"', ''), '`' + arr[0][6].replace('"', '') + '`')
            data.append(value)
            if flag == 1:
                flag = 0
                dirty_data.append(dirty_lines)
                dirty_lines = ''
    # add cleaned data to data array
    for line in dirty_data:
        arr = re.findall(reg_dirty, line)
        if len(arr) != 0:
            value = (arr[0][0].replace('"', ''), arr[0][1].replace('"', ''), arr[0][2].replace('"', ''), arr[0][3].replace('"', ''), arr[0][4].replace('"', ''), arr[0][5].replace('"', ''), '`' + arr[0][6].replace('"', '') + '`')
            data.append(value)

path = './data/comments_clean.csv'
pd.DataFrame(data).to_csv(path, header=False, index=False, encoding='utf-8')
print('Done...')