# -*- coding: utf-8 -*


import pickle
import pandas as pd
import sys

path = sys.path[0]
data = pd.read_csv(path + '/../data/user_behave.csv', encoding='utf8', sep='\t', low_memory=False)
reviews_df = pd.DataFrame(data)
reviews_df['reviewerID'] = reviews_df['reviewerID'].astype(str)
# print(reviews_df.dtypes)
# exit(0)
# # 序列化保存
# with open(path + '/../data/reviews.pkl', 'wb') as f:
#     pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
reviews_df.to_csv(path + '/../data/reviews.csv', sep='\t', index=0, encoding='utf8')

data = pd.read_csv(path + '/../data/video_feature.csv', encoding='utf8', sep='\t', low_memory=False)
meta_df = pd.DataFrame(data)

# 将asin列转换为int
meta_df['asin'] = meta_df['asin'].astype(int)
meta_df = meta_df.drop_duplicates()
# print(meta_df.dtypes)
# with open(path + '/../data/meta.pkl', 'wb') as f:
#     pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
meta_df.to_csv(path + '/../data/meta.csv', sep='\t', index=0, encoding='utf8')

