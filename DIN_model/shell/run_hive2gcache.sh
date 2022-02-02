#!/bin/bash
source /etc/profile

hive -e "
DROP TABLE IF EXISTS t_rec.target_predict_DIN_result_rank_gcache;
create table t_rec.target_predict_DIN_result_rank_gcache as
SELECT user_id,target_pred,target_rank,dt from t_rec.target_predict_DIN_result_rank
"


/home/rec/common/shell/spark-write-data-3.0  \
--genv prod --expire 172800  --table target_predict_DIN_result_rank_gcache --key user_id --cols user_id,target_pred,target_rank,dt --db t_rec --sink gcache
