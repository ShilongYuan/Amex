# Imports and Constants
import os,random 
import tqdm 
import pandas as pd
import numpy as np
import joblib
import pathlib
import tqdm
import lightgbm as lgb
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
  model_dir = "../result/lightgbm"
  sub_dir = "../result/sub"

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
    fe = f"{CFG.INPUT}/test_fe_v1.pickle"
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

# ### LGBM Params and utility functions
class SaveModelCallback:
    def __init__(self,
                 models_folder: pathlib.Path,
                 fold_id: int,
                 min_score_to_save: float,
                 every_k: int,
                 order: int = 0):
        self.min_score_to_save: float = min_score_to_save
        self.every_k: int = every_k
        self.current_score = min_score_to_save
        self.order: int = order
        self.models_folder: pathlib.Path = models_folder
        self.fold_id: int = fold_id

    def __call__(self, env):
        iteration = env.iteration
        score = env.evaluation_result_list[3][2]
        if iteration % self.every_k == 0:
            #print(f'iteration {iteration}, score={score:.05f}')
            if score > self.current_score:
                self.current_score = score 
                print(f'High Score: iteration {iteration}, score={score:.05f}')
                joblib.dump(env.model,  f'{CFG.model_dir}/lgbm_fold{self.fold_id}_seed{CFG.seed}_{score:.05f}.pkl')


def save_model(models_folder: pathlib.Path, fold_id: int, min_score_to_save: float = 0.793, every_k: int = 50):
    return SaveModelCallback(models_folder=models_folder, fold_id=fold_id, min_score_to_save=min_score_to_save, every_k=every_k)

params = {
          'objective': 'binary',
          'metric': "binary_logloss",
          'boosting': 'dart',
          'seed': CFG.seed,
          'num_leaves': 100,
          'learning_rate': 0.0075,  
          'feature_fraction': 0.20,
          'bagging_freq': 10,
          'bagging_fraction': 0.50,
          'n_jobs': -1,
          'lambda_l2': 2,
          'min_data_in_leaf': 40,
          #"histogram_pool_size":  10240
}
def lgbm_train(x, y, xt, yt,fold,
               cat_features=['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
                'D_126', 'D_63', 'D_64', 'D_66', 'D_68']):
    print("Start training")  

    lgb_train = lgb.Dataset(x, y,feature_name =[col for col in x.columns], categorical_feature = cat_features)
    lgb_valid = lgb.Dataset(xt, yt,feature_name =[col for col in x.columns], categorical_feature = cat_features)
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 11500,
        early_stopping_rounds = 100,
        verbose_eval = 50,
        valid_sets = [lgb_train, lgb_valid],  
        feval = lgb_amex_metric,
        callbacks=[save_model(models_folder=CFG.INPUT, fold_id=fold, min_score_to_save=0.7931, every_k=50)]
        )
    return model.predict(xt),model,1


# #### Metrics
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

def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True

  
# ====================================================
# Train LightGBM
# ====================================================
msgs = {}
score = 0
not_used = get_not_used()
kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed) 

if CFG.TRAIN: 
  # print(f"Number of features {len(features)}")
  oof_predictions = np.zeros(len(train))
  feature_importances = pd.DataFrame()
  not_used = [i for i in not_used if i in train.columns]  
  features = [col for col in test.columns if col not in  get_not_used()]       
  for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[[CFG.target,"S_2","customer_ID"]], train[CFG.target])):  
      _ = gc.collect()
      x, y = train[features].iloc[trn_ind], train[CFG.target].iloc[trn_ind]
      xt, yt= train[features].iloc[val_ind], train[CFG.target].iloc[val_ind]
      _ = gc.collect()      

      val_pred,model, bst = lgbm_train(x, y, xt, yt,fold)
      if fold == 0:
        feature_importances["feature"] = model.feature_name()
      feature_importances[f"importance_fold{fold}+1"] = model.feature_importance()        
      joblib.dump(model, f'{CFG.model_dir}/lgbm_fold{fold}_seed{CFG.seed}.pkl')
      amex_score = amex_metric(yt.values,val_pred) 
      msg = f"Fold {fold} amex {amex_score:.5f}"   
      oof_predictions[val_ind] = val_pred
      print(msg)
      score += amex_score   
      del x,y,xt,yt; gc.collect()
    
  oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
  oof_df.to_csv(f'{CFG.sub}/lgbm_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
  score /= CFG.n_folds
  print(f"Average amex score: {score:.5f}")       


if CFG.INFER:
  test_predictions = np.zeros(len(test))
  not_used = [i for i in not_used if i in test.columns]  
  for fold in range(CFG.n_folds):
    model = joblib.load(f"{CFG.model_dir}/lgbm_fold{fold}_seed{CFG.seed}.pkl")
    test_pred = model.predict(test[features])
    test_predictions += test_pred / CFG.n_folds 
  test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
  test_df.to_csv(f'{CFG.sub}/test_lgbm_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False) 

print('Infer finished')