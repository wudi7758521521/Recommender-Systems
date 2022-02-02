# -*- coding: utf-8 -*


import sys
import pandas as pd
import json

path = sys.path[0]


# 构建反映射函数
def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))  # 注释掉原映射函数
    m_map = dict(zip(range(len(key)), key))
    # df[col_name] = df[col_name].map(lambda x: m[x])  # 注释掉，才可使用m_map反映射函数
    return m, m_map, key


# 1.读取meta_df文件
meta_df = pd.read_csv(path + '/../data/meta.csv', encoding='utf8', sep='\t', low_memory=False)

a, asin_map, asin_key = build_map(meta_df, 'asin')  # 短视频名映射

# # 将dict类型存为json格式保存
# import json
#
# jsObj = json.dumps(asin_map)
# fileObject = open('../data/asin_m_map.json', 'w')
# fileObject.write(jsObj)
# fileObject.close()
# # exit(0)

out_result2 = pd.read_csv(path + '/../data/out_result.csv', encoding='utf8', sep='\t', low_memory=False)
# # df = df['user_id','target_pred','score','dt']  # 报错
out_result2 = out_result2[['user_id', 'target_pred', 'score', 'dt']]  # 只保留四列

# # out_result2[['target_pred']] = out_result2[['target_pred']].astype(str)  # 你说奇怪不奇怪，奇怪不奇怪，注释掉！才可以映射
# # out_result2 = out_result2['target_pred'].map(asin_map)  # 此只保留映射的一列
out_result2['target_pred'] = out_result2['target_pred'].map(asin_map)

# 根据某两列去重，保留第一次出现的，True表示直接在原数据上删除重复性
out_result2.drop_duplicates(subset=['user_id', 'target_pred'], keep='first', inplace=True)
out_result2.to_csv(path + '/../data/out_result2hive.csv', sep='\t', mode='a', index=False, encoding='utf8')

print(out_result2)
