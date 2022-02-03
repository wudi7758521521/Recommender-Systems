# -*- coding: utf-8 -*

import datetime, time
import pandas as pd
import sys

path = sys.path[0]
# 首先必须使用parse_dates将某列从字符串转换为日期格式,utf8读取数据,(date_parser不可)
data = pd.read_csv(path + '/../data/user_bhv.csv', parse_dates=['user_behave_din.visit_time'], encoding='utf8',
                   sep='\t',
                   low_memory=False)  # 必须设置为sep='\t'才可以读
# # 原数据时间列为str格式，pd.to_datetime()转换为时间格式
data['user_behave_din.visit_time'] = pd.to_datetime(data['user_behave_din.visit_time'], errors='coerce')
# 删除有问题的时间列数据格式
data.dropna(subset=['user_behave_din.visit_time'], inplace=True)
# 然后通过 '日期.timetuple()' 转换为时间元组，最后通过time.mktime()转换为时间戳
data['timestamp'] = data['user_behave_din.visit_time'].apply(lambda x: time.mktime(x.timetuple()))

# # 修改列名
data.rename(columns={'user_behave_din.video_id': 'asin', 'user_behave_din.user_id': 'reviewerID', 'timestamp': 'unixReviewTime'}, inplace=True)
# data['user_behave_din.user_id'] = data['user_behave_din.user_id'].astype(str)
# print(data.dtypes)
data = data[['reviewerID', 'asin', 'unixReviewTime']]  # 只保留三列
# print(data)
# exit(0)
# # 保存数据
data.to_csv(path + '/../data/user_behave.csv', sep='\t', index=0, encoding='utf8')

# 对video_info文件处理
data2 = pd.read_csv(path + '/../data/video_info.csv', encoding='utf8', sep='\t', low_memory=False)

data2.rename(columns={'video_info_din.id': 'asin', 'video_info_din.label': 'label', ' video_info_din.dt': 'dt'}, inplace=True)
# data2.rename(columns={'id': 'asin', 'label': 'label', ' dt': 'dt'}, inplace=True)
# data2 = data2[['asin']]

data2 = data2.drop_duplicates(subset=['asin'])
# print(data2)
# exit(0)
# data['asin'] = data['asin'].astype(str)

# # # 保存数据
data2.to_csv(path + '/../data/video_feature.csv', sep='\t', index=0, encoding='utf8')
