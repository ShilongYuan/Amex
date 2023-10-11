运行步骤
1 下载数据集 （https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format），PS：采用处理过的数据集
2 运行code/fe_process.py，该步骤主要目的是生成parquet特征文件，运行需要一定时间
3 运行code/lgb.py，这个代码训练lgb
4运行code/xgb.py，这个代码训练xgb
5 运行code/infer.ipynb，得到融合结果
7 提交result/sub/submission.csv文件到kaggle

P.S: 因为lgb训练时间过长，本复盘提供训练生成文件
一些依赖的包需要自行安装一下
pandas
numpy
lightgbm
pyarrow
pickle
tdqm

运行需要内存较大，建议64G内存
