# a_dict = {2121:3,'b':4}
# if 2121 in a_dict:
#     c = a_dict[2121]
#     print(c)
# else:
#     print('21b')
#
import random
import tensorflow as tf
# a = [1, 2, 3, 4, 5, 6]
# d = random.sample(a, 5)
# print(d)
# b = a * 2
# # print(b)
# c = []
# for i in range(len(a)):
# # c.append([1, 2, i for i in range(len(a))])
#     c.append([1, 2, 3])
# # print(c)
#
# d = []
# # [d.append(1, 2, i) for i in [[1], [2], [3]]]
# [d.append(i) for i in [[1], [2], [3]]]
# # [d.extend(i) for i in [[1], [2], [3]]]
# # #
# print(d)
#
import random
import numpy as np
#
#
# target_list = [i for i in range(10)]
# target_list_pred1 = []
# # print(target_list)
# b = np.random.choice(target_list, 6, replace=False)

# print(b)

# for i in range(len(target_list)):
#     target1 = random.sample(target_list, 1)
#     target_list_pred1.append(target1)
#     # print(target1[0])
#     # exit(0)
#     if target1[0] in target_list:
#         target_list.remove(target1[0])
#
# # print(target1)
# #     print(i)
# # exit(0)
# print(target_list_pred1)

# 反映射模块
# import sys
# import pandas as pd
# import json
# # from remap import build_map
#
#
# def build_map(df, col_name):
#     """
#     制作一个映射，键为列名，值为序列数字
#     :param df: reviews_df / meta_df
#     :param col_name: 列名
#     :return: 字典，键
#     """
#     key = sorted(df[col_name].unique().tolist())
#     m = dict(zip(key, range(len(key))))
#     m_map = dict(zip(range(len(key)), key))
#     # df[col_name] = df[col_name].map(lambda x: m[x])  # 注释掉，才可使用m_map反映射函数
#     return m, m_map, key
#
#
# #
# #
# path = sys.path[0]
# meta_df = pd.read_csv(path + '/../data/meta.csv', encoding='utf8', sep='\t', low_memory=False)
# # reviews_df = pd.read_csv(path + '/../data/reviews.csv', encoding='utf8', sep='\t', low_memory=False)
# # # 修改列名
# # reviews_df.rename(columns={'user_behave_din.video_id': 'asin', 'user_behave_din.user_id': 'reviewerID'}, inplace=True)
# # reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]  # 只保留三列
#
# # print(meta_df)
# # exit(0)
#
# a, asin_map, asin_key = build_map(meta_df, 'asin')  # 短视频名映射
# # b, revi_map, revi_key = build_map(reviews_df, 'reviewerID')  # 用户名映射
# # print(revi_map)
# # # # print(asin_key)
# # exit(0)
#
# # # 将dict类型存为json格式保存
# # import json
# #
# # jsObj = json.dumps(revi_map)
# # fileObject = open('../data/revi_map.json', 'w')
# # fileObject.write(jsObj)
# # fileObject.close()
# # exit(0)
# #
# out_result2 = pd.read_csv(path + '/../data/out_result.csv', encoding='utf8', sep='\t', low_memory=False)
# # print(out_result2)
# # exit(0)
# # reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
# # out_result2['user_id'] = out_result2['user_id'].map(revi_map)
# out_result2['target_pred1'] = out_result2['target_pred1'].map(asin_map)
# # out_result2 = out_result2['target_pred1']
#
# out_result2.to_csv(path + '/../data/out_result_12.csv', sep='\t', mode='a', index=False, encoding='utf8')
#
# print(out_result2)
# # exit(0)

# a = [2]
# for i in range(1, len(a)):
#     print(i)
#     if i == len(a) - 1:
#         print(i)
#         print('aaa')


# a = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8], ['a', 'b', 'c'], [1, 2, 3, 4, 5, 6]]
# target = []
# for i in a:
#     if len(i) < 7:
#         b = random.sample(i, len(i))
#         target.append(b)
#     else:
#         b = random.sample(i, 7)
#         target.append(b)
#
# print(target)
# f = [7, 9, 3, 1, 11, 4, 2]
# p = []
# for i in target:
#     # print(i)
#     # print(len(i))
#     for g in f:
#         q = [g] * len(i)
#         p.extend(q)
# u = []
# [u.extend(i) for i in target]  # 去掉列表嵌套
# # print(u)
# print(len(u))
# # print(p)
# print(len(p))

#
# a = 1
# print([a] * 3)


# a = [[1, 2, 3, 4, 5], [1, 2, 3], [1, 2, 3, 4]]
# for i in range(len(a)):
#     print(len(a[i]))

#
# a = [1, 2, 3, 4, 5,6]
# b = a[0: len(a)-1]
# print(b)


# for i in range(10):
#     print(i)
import random


