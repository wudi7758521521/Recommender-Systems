#!/bin/bash
source /etc/profile

# 设置shell脚本时间，在sql语句中约束dt字段获取数据
yesterday=$(date -d "-1 day" +%Y-%m-%d)
fivedaysago=$(date -d "-20 day" +%Y-%m-%d)
echo $yesterday $fivedaysago


# hive -f表示从文件中运行sql语句
hive --hivevar enddate=$yesterday \
--hivevar begindate=$fivedaysago \
-f /home/rec/azkaban-jobs/Ankai/DIN_model/hql/hql00.sql


