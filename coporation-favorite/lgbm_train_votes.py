#Based on Ceshine's LGBM starter script https://www.kaggle.com/ceshine/lgbm-starter

from datetime import date, timedelta
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "last_week_2017": get_timespan(df_2017, t2017, 7, 1).values.ravel(),
        "lastlast_week_2017": get_timespan(df_2017, t2017, 14, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "promo_3_2017": get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
        "promo_7_2017": get_timespan(promo_2017, t2017, 7, 7).sum(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
        "oil_1_2017": get_timespan(oil_2017, t2017, 1, 1).values.ravel(),
        "oil_3_2017": get_timespan(oil_2017, t2017, 3, 3).mean(axis=1).values,
        "oil_7_2017": get_timespan(oil_2017, t2017, 7, 7).mean(axis=1).values,
        "diff_1_2017": get_timespan(diff_2017, t2017, 1, 1).values.ravel(),
        "diff_3_2017": get_timespan(diff_2017, t2017, 3, 3).mean(axis=1).values,
        "diff_7_2017": get_timespan(diff_2017, t2017, 7, 7).mean(axis=1).values,
        "diff_14_2017": get_timespan(diff_2017, t2017, 14, 14).mean(axis=1).values,
        "diff_30_2017": get_timespan(diff_2017, t2017, 30, 30).mean(axis=1).values
    })
    for i in range(7):
        X['mean_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 4, freq='D').mean(axis=1).values
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

def train(X_train, y_train, X_val, y_val, X_test, params, num_weeks = 10):
    MAX_ROUNDS = 500
    val_pred = []
    test_pred = []
    for i in range(16):
        print("=" * 50)
        print("Step %d" % (i+1))
        print("=" * 50)
        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            weight=pd.concat([items["perishable"]] * num_weeks) * 0.25 + 1
        )
        dval = lgb.Dataset(
            X_val, label=y_val[:, i], reference=dtrain,
            weight=items["perishable"] * 0.25 + 1)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
        )
        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    print("Validation mse:", mean_squared_error(
        y_val, np.array(val_pred).transpose()))
    return val_pred, test_pred


###############################################
oils = pd.read_csv('imputed_oils.csv', parse_dates = ['date'])

df_train = pd.read_csv(
    os.path.join('input','train.csv'), usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    os.path.join('input', 'test.csv'), usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
)

items = pd.read_csv(
    os.path.join('input', 'items.csv'),
).set_index("item_nbr")

oils = oils.drop(['Unnamed: 0'], axis = 1)
oils["oil_price_diff"] = oils["oil_price"].diff().fillna(0.0)

df_2017 = df_train[df_train['date'] >= pd.datetime(2016, 12, 31) ]
df_2017 = pd.merge(df_2017, oils, how = 'left', on='date')
df_2017 = df_2017.set_index(["store_nbr", "item_nbr", "date"])

df_test = pd.merge(df_test, oils, how = 'left', on = 'date')
df_test = df_test.set_index(
    ['store_nbr', 'item_nbr', 'date']
)
#check distribution of oil prices differences
import matplotlib.pyplot as plt
df_2017["oil_price_diff"].hist()
plt.show()

#pivot oil prices
oil_2017_train = df_2017[["oil_price"]].unstack(
        level=-1).fillna(False)
oil_2017_train.columns = oil_2017_train.columns.get_level_values(1)

oil_2017_test = df_test[['oil_price']].unstack(level = -1).fillna(False)
oil_2017_test.columns = oil_2017_test.columns.get_level_values(1)
oil_2017_test = oil_2017_test.reindex(oil_2017_train.index).fillna(df_test['oil_price'].mean())

oil_2017 = pd.concat([oil_2017_train, oil_2017_test], axis=1)
del oil_2017_train, oil_2017_test

#pivot oil price differences
diff_2017_train = df_2017[['oil_price_diff']].unstack(level = -1).fillna(False)
diff_2017_train.columns = diff_2017_train.columns.get_level_values(1)

diff_2017_test = df_test[['oil_price_diff']].unstack(level = -1).fillna(False)
diff_2017_test.columns = diff_2017_test.columns.get_level_values(1)
diff_2017_test = diff_2017_test.reindex(diff_2017_train.index).fillna(0)

diff_2017 = pd.concat([diff_2017_train, diff_2017_test], axis = 1)
del diff_2017_train, diff_2017_test

promo_2017_train = df_2017[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)

promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)

promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

print("Training and predicting models...")
params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 8
}
num_weeks = 8
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)
df_preds = None
MSEs = []
for d in range(7):
    t2017 = date(2017, 5, 30) + timedelta(days = d)
    X_l, y_l = [], []
    for i in range(num_weeks):
        delta = timedelta(days=7 * i)
        X_tmp, y_tmp = prepare_dataset(
            t2017 + delta
        )
        X_l.append(X_tmp)
        y_l.append(y_tmp)
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    X_val, y_val = prepare_dataset(date(2017, 7, 25 + d))
    val_pred, test_pred = train(X_train, y_train, X_val, y_val, X_test, params, num_weeks)
    mse = mean_squared_error(y_val, np.array(val_pred).transpose())
    MSEs.append(mse)
    print("Validation mse:", mse)
    print("Making submission...")
    y_test = np.array(test_pred).transpose()
    if df_preds is None:
        df_preds = pd.DataFrame(
            y_test, index=df_2017.index,
            columns=pd.date_range("2017-08-16", periods=16)
        ).stack().to_frame("unit_sales_"+str(d))
    else:
        p = pd.DataFrame(
            y_test, index=df_2017.index,
            columns=pd.date_range("2017-08-16", periods=16)
            ).stack().to_frame("unit_sales_" + str(d))
        df_preds = pd.concat([df_preds, p], axis = 1)

print(sum(MSEs)/len(MSEs))
print(MSEs)
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
df_preds['unit_sales'] = df_preds.mean(axis = 1)


submission = df_test[["id"]].join(df_preds[['unit_sales']], how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb_mean_vote.csv', float_format='%.4f', index=None)



    
