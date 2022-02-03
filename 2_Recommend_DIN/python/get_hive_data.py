# _*_coding:utf-8 _*_

from pyhive import hive
import pandas as pd
import sys

path = sys.path[0]
# conn表示连接数据库，对数据的操作需要通过cursor来实现。
conn = hive.Connection(host='10.112.183.31', port=10000, auth='KERBEROS', kerberos_service_name="hive")
cursor = conn.cursor()
# cursor.execute()表示执行数据库操作
cursor.execute("select * from tmp.user_behave_DIN limit 100")

columns = [col[0] for col in cursor.description]  # 字段名带表明
result = [dict(zip(columns, row)) for row in cursor.fetchall()]
data = pd.DataFrame(result)
# data_item.columns = columns

# data = cursor.fetchall()
print(data)
# df = pd.read_sql("select * from tmp.user_behave_DIN limit 100", conn)   # 此处sql语句结尾出不带;号

# df.to_csv(path + '/../data/user_df.csv', sep='\t', index=0, encoding='utf8')
# print(df)