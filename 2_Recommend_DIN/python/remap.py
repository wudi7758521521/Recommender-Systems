# -*- coding: utf-8 -*
#
import re
import random
import pickle
import sys

import numpy as np
import pandas as pd


def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    # m_map = dict(zip(range(len(key)), key))  # 反映射
    # df[col_name] = df[col_name].map(lambda x: m[x])  # 注释掉，才可使用m_map反映射函数
    return m, key


path = sys.path[0]
# 1.读取reviews
# reviews_df = pd.read_pickle(path+'/../data/reviews.pkl')
reviews_df = pd.read_csv(path + '/../data/reviews.csv', encoding='utf8', sep='\t', low_memory=False)
# 修改列名
# reviews_df.rename(columns={'user_behave_din.video_id': 'asin', 'user_behave_din.user_id': 'reviewerID'}, inplace=True)
# reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]  # 只保留三列
# reviews_df['reviewerID'].astype(str)
# print(reviews_df.dtypes)
# # print(reviews_df)
# exit(0)
# 2.读取meta
# meta_df = pd.read_pickle(path+'/../data/meta.pkl')
meta_df = pd.read_csv(path + '/../data/meta.csv', encoding='utf8', sep='\t', low_memory=False)
# print(meta_df)
# exit(0)
# meta_df = meta_df[['asin']]  # 只保留一列

# # # 3.读取target
# target = pd.read_csv(path + '/../data/target_category.csv', sep='\t')
# # # target = pd.read_csv(path + '/../data/target_category_pred_bucket.csv', sep='\t')
# #
# target = pd.DataFrame(target)
# # # print(target)
# # # exit(0)

# meta_df文件的物品ID映射(asin)
asin_map, asin_key = build_map(meta_df, 'asin')
label_map, label_key = build_map(meta_df, 'label')
# print(label_map)
# # print(asin_key)
# exit(0)

# reviews_df文件的用户ID映射(reviewerID)
revi_map, revi_key = build_map(reviews_df, 'reviewerID')
# print(revi_map)
# # print(asin_key)
# exit(0)

# # 将dict类型存为json格式保存
# import json
#
# # jsObj = json.dumps(revi_map)
# jsObj1 = json.dumps(asin_map,ensure_ascii=False)
# # fileObject = open('../data/revi_map.json', 'w', encoding='utf-8')
# fileObject1 = open('../data/asin_map_id2num.json', 'w', encoding='utf-8')
# # fileObject.write(jsObj)
# fileObject1.write(jsObj1)
# # fileObject.close()
# fileObject1.close()
# exit(0)


user_count, item_count, example_count = \
    len(revi_map), len(meta_df), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\texample_count: %d' % (user_count, item_count, example_count))

# 按物品id排序，并重置索引
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
# meta_df.to_csv(path + '/../data/meta_map.csv', sep='\t', mode='a', index=False, encoding='utf8')

# print(meta_df)
# exit(0)


reviews_df = pd.DataFrame(reviews_df)
# reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
reviews_df['asin'] = reviews_df['asin'].map(asin_map)
meta_df['asin'] = meta_df['asin'].map(asin_map)
meta_df.to_csv(path + '/../data/meta_map.csv', sep='\t', index=0, encoding='utf8')
# ## target文件的video_id根据meta文件映射的video_id进行一一对应，确保同一个video_id映射为同一个数值
# target['rec_video_din_category_pred.video_id1'] = target['rec_video_din_category_pred.video_id1'].map(asin_map)
# target['rec_video_din_category_pred.video_id2'] = target['rec_video_din_category_pred.video_id2'].map(asin_map)
# # target = target[['rec_video_din_category_pred_bucket.video_id1', 'rec_video_din_category_pred_bucket.category','rec_video_din_category_pred_bucket.video_id2', 'rec_video_din_category_pred_bucket.dt']]  # 只保留三列
# target = target[['rec_video_din_category_pred.video_id1', 'rec_video_din_category_pred.category',
#                  'rec_video_din_category_pred.video_id2', 'rec_video_din_category_pred.dt']]  # 只保留三列
# # print(target)
# # exit(0)
# target_map = target.to_csv(path + '/../data/target_map.csv', sep='\t', encoding='utf8', index=False)

reviews_df = reviews_df.sort_values('asin')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# isnan()判断是否是空值,是空值drop掉
reviews_df = reviews_df.drop(reviews_df[np.isnan(reviews_df['asin'])].index, inplace=False)
# reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])

pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 禁用科学计算法
reviews_df = reviews_df.reset_index(drop=True)  # 重置索引
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# 根据'reviewerID', 'asin'两列去重
reviews_df = reviews_df.drop_duplicates(subset=['reviewerID', 'asin'])
reviews_df['asin'] = reviews_df['asin'].astype(int)
# print(reviews_df)
# exit(0)
# # 保存所需数据为pkl文件
# with open(path + '/../data/remap.pkl', 'wb') as f:
#     pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
#     pickle.dump((user_count, item_count, example_count), f, pickle.HIGHEST_PROTOCOL)
#     pickle.dump((asin_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
reviews_df.to_csv(path + '/../data/remap.csv', sep='\t', index=0, encoding='utf8')
# f = open('./remap.pkl', 'rb')
# data = pickle.load(f)
# # 判断有无重复值，False表示当前位置和上个位置无重复，True表示当前位置和上个位置有重复
# # data = data['reviewerID'].duplicated()
# # data = data.duplicated(subset='reviewerID')
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
# data.to_csv('./remap.csv', sep='\t', index=0, encoding='utf8')
