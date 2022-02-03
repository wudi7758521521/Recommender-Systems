#!/bin/bash
source /etc/profile  # 保存配置

output_file="/home/rec/azkaban-jobs/Ankai/DIN_model/data/out_result2hive.csv"

hive -e "
CREATE EXTERNAL TABLE if not exists tmp.target_predict_DIN_result (
  user_id string,
  video_hist string,
  target_pred string,
  score double,
  dt string
  )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE;
"

exitCode=$?
if [ $exitCode -ne 0 ];then
         echo "[ERROR]  execute mmoe_pred_model compute failed!"
         exit $exitCode
fi


hive -e "LOAD DATA LOCAL INPATH '${output_file}'  overwrite INTO TABLE tmp.target_predict_DIN_result;"

rm $output_file

hive -e "
DROP TABLE IF EXISTS t_rec.target_predict_DIN_result_rank;
create table t_rec.target_predict_DIN_result_rank as
select *,row_number() over(partition by user_id order by score desc) as target_rank
from tmp.target_predict_DIN_result;"

# 删掉保存的文件
DATA="/home/rec/azkaban-jobs/Ankai/DIN_model/data"
rm $DATA/user_bhv.csv
rm $DATA/video_info.csv

rm $DATA/user_behave.csv
rm $DATA/video_fea.csv

rm $DATA/reviews.csv
rm $DATA/meta.csv
rm $DATA/target_category.csv

rm $DATA/target_map.csv
rm $DATA/remap.csv

rm $DATA/out_result.csv
