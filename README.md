# Amex

运行步骤
1 下载[数据集](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format)，PS：采用处理过的数据集

2 运行code/fe_process.py，该步骤主要目的是生成parquet特征文件，运行需要一定时间

3 运行code/lgb.py，这个代码训练lgb

4运行code/xgb.py，这个代码训练xgb

5 运行code/infer.ipynb，得到融合结果

依赖的包:

pandas

numpy

lightgbm

pyarrow

pickle

tdqm
