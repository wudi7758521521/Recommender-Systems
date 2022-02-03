#!/bin/bash
source /home/rec/python_env/bin/activate

PYTHONPATH="/home/rec/azkaban-jobs/Ankai/DIN_model/python"

# 环境中有python2和python3
python3 $PYTHONPATH/unixReviewTime.py

python3 $PYTHONPATH/convert.py

python3 $PYTHONPATH/remap.py

python3 $PYTHONPATH/train.py




