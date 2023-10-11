# Imports and Constants
import os,random 
import tqdm 
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import pathlib
import tqdm

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
  TRAIN = True 
  INFER = True
  n_folds = 5
  target ='target'
  DEBUG= False 
  ADD_CAT = True
  ADD_LAG = True 
  ADD_DIFF =  [1, 2]
  ADD_MIDDLE = True
  INPUT = "../data"
  model_dir = "../result/catboost"
  sub_dir = "../result/sub"


path = f'{CFG.INPUT}'  


# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.seed)    


def get_not_used():  
  return ['row_id', 'customer_ID', 'target', 'cid', 'S_2','D_103','D_139']    

# ====================================================
#   数据加载
# ====================================================
if CFG.TRAIN:
  fe = f"{CFG.INPUT}/train_fe_v1.pickle"
  if os.path.exists(fe):
    train = pd.read_pickle(fe) 
  print(train.shape)
  train.head()

if CFG.DEBUG:
  train = train.sample(n=2000, random_state=42).reset_index(drop=True)
  features = [col for col in train.columns if col not in  get_not_used()]
  
if CFG.INFER:
    fe = f"{CFG.INPUT}/test_fe_v16.pickle"
    if os.path.exists(fe):
      test = pd.read_pickle(fe) 
    print(test.shape)
    test.head()
_ = gc.collect()

def do_miss_nan(df):
    # Impute missing values
    df.fillna(value=-1, inplace=True)
    # Replace inf with zeros 
    df.replace([np.inf, -np.inf], -1, inplace=True)
    # Reduce memory
    for c in df.columns:
      if c in get_not_used(): continue
      if str( df[c].dtype )=='int64':
          df[c] = df[c].astype('int32')
      if str(df[c].dtype )=='float64':
          df[c] = df[c].astype('float32')
    return df

train = do_miss_nan(train)
test = do_miss_nan(test)

# ====================================================
# 模型构建
# ====================================================

params = {
        #'booster': 'dart',
        'objective': 'binary:logistic', 
        'tree_method': 'gpu_hist', 
        'max_depth': 8,
        'subsample':0.88,
        'colsample_bytree': 0.5,
        'gamma':1.5,
        'min_child_weight':8,
        'lambda':70,
        'eta':0.02, 
}

def xgb_train(x, y, xt, yt,_params= params):
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    dtrain = xgb.DMatrix(data=x, label=y)
    dvalid = xgb.DMatrix(data=xt, label=yt)
 
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(_params, dtrain=dtrain,
                num_boost_round=4000,evals=watchlist,
                early_stopping_rounds=600, feval=xgb_amex, maximize=True,
                verbose_eval=100)
    print('best ntree_limit:', bst.best_ntree_limit)
    print('best score:', bst.best_score)
    return bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit)), bst

# Metrics
def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())
# Created by https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

# we still need the official metric since the faster version above is slightly off
def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)


  
# ====================================================
# Train XG_BOOST
# ====================================================
msgs = {}
score = 0
kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed) 

def train_fn(i,x,y,xt,yt,_params= params): 
    print("Start training")  
    yp, bst = xgb_train(x, y, xt, yt,_params)
    bst.save_model(f'{CFG.model_dir}/xgb_{i}.json')
    amex_score =  amex_metric(yt.values,yp) 
    return amex_score, bst,yp



if CFG.TRAIN: 
    oof_predictions = np.zeros(len(train)) 
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    features = [col for col in train.columns if col not in  get_not_used()]
 
    not_used = get_not_used()
    not_used = [i for i in not_used if i in train.columns]
    msgs = {}
    score = 0 
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        x, y = train[features].iloc[trn_ind], train[CFG.target].iloc[trn_ind]
        xt, yt= train[features].iloc[val_ind], train[CFG.target].iloc[val_ind]
        amex_score, model,yp = train_fn(fold,x,y,xt,yt)
        print(f'Our fold {fold} CV score is {amex_score}')
        oof_predictions[val_ind] = yp
        score += amex_score
        del x,y,xt,yt,yp; gc.collect()
        torch.cuda.empty_cache()
    score /= CFG.n_folds
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    #display(oof_df.head())
    oof_df.to_csv(f'{CFG.sub_dir}/xg_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)   
    print(f"Average amex score: {score:.4f}")
      
if CFG.INFER:
  test_predictions = np.zeros(len(test))
  yps = []
  not_used = [i for i in not_used if i in test.columns]
  yp=0
  test_predictions = np.zeros(len(test))
  for fold  in range(CFG.n_folds):
    bst = xgb.Booster()           
    bst.load_model(f"{CFG.model_dir}/xgb_{fold}.json") 
    dx = xgb.DMatrix(test.drop(not_used, axis=1))
    print('best ntree_limit:', bst.best_ntree_limit)
    test_predictions += bst.predict(dx, iteration_range=(0,bst.best_ntree_limit))/CFG.n_folds 
  test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
  test_df.to_csv(f'{CFG.sub_dir}/test_xg_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False) 
print('Infer finished')


