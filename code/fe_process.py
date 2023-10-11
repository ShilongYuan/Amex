# Imports and Constants
import os,random 
import tqdm 
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np
import joblib
import pathlib
import tqdm
import time 
import time 
import torch, gc 
torch.cuda.empty_cache()
gc.collect()    
from sklearn.model_selection import StratifiedKFold
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class CFG:
  seed = 42
  INPUT = "../data/amex-data-integer-dtypes-parquet-format"
  TRAIN = True 
  INFER = True
  n_folds = 5
  target ='target'
  DEBUG= True 
  ADD_CAT = True
  ADD_LAG = True 
  ADD_DIFF =  [1, 2]
  ADD_MIDDLE = True
  output_dir = "../data"

path = f'{CFG.INPUT}'  


# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.seed)    



features_avg = ['S_2_wk','B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18',
                'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42',
                'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 
                'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 
                'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123',
                'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145',
                'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 
                'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']

# Feature Engineering on credit risk
spend_p=[ 'S_3',  'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_18', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
balance_p = ['B_1', 'B_2', 'B_3',  'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15',  'B_17', 'B_18',  'B_21',   'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28',  'B_36', 'B_37',  'B_40',    ]
payment_p = ['P_2', 'P_3', 'P_4']
delq = ['D_39',
                'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 
                'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141', 'D_144',
                'D_145']  
cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
            'D_126', 'D_63', 'D_64', 'D_66', 'D_68']     

cat_cols_avg = [col for col in cat_cols if col in features_avg]
g_num_cols = []

# ====================================================
#              Feature Engineering
# ====================================================
def process_data(df):
    df,dgs = preprocess(df) 
    df = df.drop_duplicates('customer_ID',keep='last')
    for dg in dgs:
        df = df.merge(dg, on='customer_ID', how='left')
        # drop specific non impactful cols 
    del dgs; gc.collect()    
             
    diff_cols = [col for col in df.columns if col.endswith('_diff')]
    df = df.drop(diff_cols,axis=1)
    print(f"All stats merged {len(df.columns)}")   
  
    math_col = globals()['g_num_cols']
    # More Lag Features
    for col in spend_p+payment_p+balance_p:
        for col_2 in ['min','max']: 
          if f"{col}_{col_2}" in df.columns:
              df[f'{col}_{col_2}_lag_sub'] = df[f"{col}_{col_2}"] - df[col]
              df[f'{col}_{col_2}_lag_div'] = df[f"{col}_{col_2}"] / df[col] 
    print("Added more lags")

    # add More custom features
    df["P2B9"] = df["P_2"] / df["B_9"] 
    math_col = globals()['g_num_cols']
    for pcol in math_col:
      if pcol+"_mean" in df.columns:  
        df[f'{pcol}-mean'] = df[pcol] - df[pcol+"_mean"]  
        df[f'{pcol}-div-mean'] = df[pcol] /df[pcol+"_mean"]
      if (pcol+"_min" in df.columns) and (pcol+"_max" in df.columns):  
        df[f'{pcol}_min_div_max'] = df[pcol+"_min"] / df[pcol+"_max"]  
        df[f'{pcol}_min-max'] = df[pcol+"_min"] - df[pcol+"_max"]
    print(f"Addding col-mean {len(df.columns)} cols {math_col}")     


    # Dropping Sum
    drop_col = [col for  col in df.columns if  (("sum" in col))]
    print(f"Dropping {drop_col}")
    df=df.drop(drop_col,axis=1)   

    print(f"Addding col-mean + custom features {len(features_avg)} cols {globals()['g_num_cols']}")    
    return df

   
def preprocess(df):
    df['row_id'] = np.arange(df.shape[0])
    not_used = get_not_used()
    # Drop cols https://www.kaggle.com/code/raddar/redundant-features-amex/notebook
    df=df.drop(["D_103","D_139"],axis=1)
    num_cols = [col for col in df.columns if col not in cat_cols+not_used]   

    globals()['g_num_cols'] = num_cols
    for col in df.columns:
        if col not in not_used+cat_cols:
           df[col] = df[col].astype('float32').round(decimals=2).astype('float16') 
    print(f"Starting fe [{len(df.columns)}]") 
    dgs=add_stats_step(df, num_cols) # 对数值型变量添加统计特征

    train_stat = df.groupby("customer_ID")[spend_p+payment_p+delq+balance_p].agg('sum')
    train_stat.columns = [x+'_sum' for x in train_stat.columns]
    print(train_stat.columns)
    train_stat.reset_index(inplace = True)    
    dgs.append(train_stat)
    del train_stat; gc.collect() 
    print(f"Stats Sum calc [{len(df.columns)}]")       
 
    # Add P-S features
    df["P_SUM"] = df[payment_p].sum(axis=1) 
    df["S_SUM"] = df[spend_p].sum(axis=1) 
    df["B_SUM"] = df[balance_p].sum(axis=1)
    df["P-S"] = df.P_SUM - df.S_SUM       
    df["P-B"] = df.P_SUM - df.B_SUM
    df=df.drop(["S_SUM","P_SUM","B_SUM"],axis=1)
    print(f"P-S feature added")      


    # Add Lag Columns 
    if CFG.ADD_LAG:
      train_num_agg = df.groupby("customer_ID")[num_cols].agg(['first', 'last']) #payment_p+balance_p+spend_p
      train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
      train_num_agg.reset_index(inplace = True) 
      for col in train_num_agg:
        if 'last' in col and col.replace('last', 'first') in train_num_agg:
                    train_num_agg[col + '_lag_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'first')]
                    train_num_agg[col + '_lag_div'] = train_num_agg[col] / train_num_agg[col.replace('last', 'first')]            
      train_num_agg.drop([col for col in train_num_agg.columns if "last" in col],axis=1, inplace=True)
      dgs.append(train_num_agg)
      del train_num_agg
      print(f"Computing diff 1 features ,curr cols [{len(df.columns)}]") 
      dff_cols =  payment_p+balance_p+spend_p+delq ## Replace with num_cols
      

      # add diff features
      for pdf in CFG.ADD_DIFF:
        train_diff = get_difference(df, dff_cols,period=pdf)
        print(f"Computing Diff {pdf} ,curr cols [{ train_diff.columns}]") 
        dgs.append(train_diff)    
        del train_diff; gc.collect()             
    
    # compute "after pay" features
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
    # 
    df['S_2'] = pd.to_datetime(df['S_2'])
    df['cid'], _ = df.customer_ID.factorize()    

    # Add sundays count as a feature 
    s2_count = df[df.S_2.dt.dayofweek == 6].groupby("customer_ID")['S_2'].agg(['count']) 
    s2_count.columns = ['S_2_Sun_Count']
    s2_count.reset_index(inplace = True)     
    dgs.append(s2_count)
    print(f"sundays count added and calculated [{len(s2_count.columns)}]") 

    # Add week of the month correlation 查看标准差
    df['S_2_wk'] =  df['S_2'].dt.week
    s2_count = df.groupby("customer_ID")['S_2_wk'].agg(['std'])  
    s2_count.columns = ['S_2_wk_std']
    s2_count.reset_index(inplace = True)     
    dgs.append(s2_count)
    df=df.drop(["S_2_wk"],axis=1 )
    print(f"sundays count added and calculated [{len(s2_count.columns)}]")        
    del s2_count; gc.collect()     


    if CFG.ADD_CAT:  # 对类别型变量做统计特征
      train_cat_agg = df.groupby("customer_ID")[cat_cols].agg(['count', 'nunique', 'std','first']) 
      train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
      train_cat_agg.reset_index(inplace = True)     
      dgs.append(train_cat_agg)
      del train_cat_agg; gc.collect() 
      train_cat_mean = df.groupby("customer_ID")[cat_cols_avg].agg(['mean']) 
      train_cat_mean.columns = ['_'.join(x) for x in train_cat_mean.columns]
      train_cat_mean.reset_index(inplace = True)    
      print(f"Added cat mean cols [{train_cat_mean.columns}]")   
      dgs.append(train_cat_mean)
      del train_cat_mean; gc.collect() 
      print(f"CAT features added {len(df.columns)}") 

    # Add s2 count as a feature ( Number of spends)
    s2_count = df.groupby("customer_ID")['S_2'].agg(['count']) 
    s2_count.columns = ['S_2_Count']
    s2_count.reset_index(inplace = True)    
    df = df.merge(s2_count, on='customer_ID', how='inner')
    print(f"Stats added and calculated [{len(s2_count.columns)}]")    
    del s2_count; gc.collect() 

    # 采集客户中间状态
    if CFG.ADD_MIDDLE:
      df_middle = df[df.S_2_Count > 2].groupby(['customer_ID'])[balance_p+payment_p+delq+spend_p].apply(lambda x: x.iloc[(len(x)+1)//2])   
      df_middle.columns = [x+'_mid' for x in df_middle.columns]  
      dgs.append(df_middle) 
      print(f"Mid Cols added [{len(df_middle.columns)}]")    
      del df_middle; gc.collect() 
      
    # restore the original row order by sorting row_id
    df = df.sort_values('row_id')
    df = df.drop(['row_id'],axis=1)

    return df, dgs


def get_not_used():  
  return ['row_id', 'customer_ID', 'target', 'cid', 'S_2','D_103','D_139']    

def add_stats_step(df, cols):
    n = 50
    dgs = []
    for i in range(0,len(cols),n):
        s = i
        e = min(s+n, len(cols))
        dg = add_stats_one_shot(df, cols[s:e])
        dgs.append(dg)
    return dgs

stats = ['mean', 'min', 'max','std']  # 进行如下聚合运算
def add_stats_one_shot(df, cols):
    
    dg = df.groupby('customer_ID').agg({col:stats for col in cols})
    out_cols = []
    for col in cols:
        out_cols.extend([f'{col}_{s}' for s in stats])
    dg.columns = out_cols
    dg = dg.reset_index()
    return dg

# Get the difference
def get_difference(data, num_features,period=1): 
    df1 = []
    customer_ids = []
    for customer_id, df in  data.groupby(['customer_ID']):
        # Get the differences
        diff_df1 = df[num_features].diff(period).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + f'_diff{period}' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1



# ====================================================
#   数据保存
# ====================================================
train = pd.read_parquet(f'{CFG.INPUT}/train.parquet') 
train = process_data(train) 
trainl = pd.read_csv(f'{CFG.INPUT}/train_labels.csv')
trainl.target = trainl.target.astype('int8')  
train = train.merge(trainl, on='customer_ID', how='left')
train.to_pickle(f"{CFG.output_dir}/train_fe_v1.pickle")
print("Saving train FE to file") 


test = pd.read_parquet(f'{CFG.INPUT}/test.parquet') 
test = process_data(test) 
test.to_pickle(f"{CFG.output_dir}/test_fe_v1.pickle")
print("Saving test FE to file")   
print('FE finished')






