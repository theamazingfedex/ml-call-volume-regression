from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime

epoch = datetime.utcfromtimestamp(0)
df = pd.read_csv('./data/2014-2015 Daily Calls by Campaign.csv')
msk = np.random.rand(len(df)) < 0.8
daydict = {
    "Sunday": 1,
    "Monday": 2,
    "Tuesday": 3,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 6,
    "Saturday": 7
    }

def to_millis(dt):
  return (dt - epoch).total_seconds() * 1000.0
df = df.drop([
  # 'CallDate',
  'Calls Outbound',
  'Calls Inbound Handled',
  'Calls Outbound Handled',
  'Calls Self-Served',
  'Calls Abandoned',
  'Calls Presented',
  'Handle Time - Inbound Average',
  'Handle Time - Outbound Average',
  'Avg.Wait Time',
  'Avg. Wait Time - Abandoned',
  'Avg. Speed To Answer',
  'Avg. Speed To Answer With Transfer'
  ], axis=1)
print df.columns

df_train =  df[msk]
df_train = df_train.dropna(how='any')

df_test = df[~msk]
df_test = df_test.dropna(how='any')

Xt = DataFrame()
yt = DataFrame()

Xt['CallDate'] = df_test[['CallDate']].applymap(lambda x: to_millis(pd.to_datetime(x)))
Xt['DayOfWeek'] = df_test[['DayOfWeek']].applymap(lambda x: daydict[x])
Xt['labels'] = df_test[['Calls Inbound']]
Xt = Xt[np.isfinite(Xt['DayOfWeek'])]

yt = Xt['labels']
Xt = Xt.drop(['labels'], axis=1)

X = DataFrame()
y = DataFrame()
X['CallDate'] = df_train[['CallDate']].applymap(lambda x: to_millis(pd.to_datetime(x)))
X['DayOfWeek'] = df_train[['DayOfWeek']].applymap(lambda x: daydict[x])

X['labels'] = df_train[['Calls Inbound']]
X = X[np.isfinite(X['DayOfWeek'])]
y = X['labels']
X = X.drop(['labels'], axis=1)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

# clf = SVR(kernel='sigmoid', C=10000.0, gamma=1.0)
clf1 = LinearRegression()
clf = AdaBoostRegressor(learning_rate=0.1, n_estimators=100)

# print df_train
# clf = GaussianNB()
# df_train.dropna(how='any')
print "X.shape = ", X.shape[1]
clf.fit(X, y)

thetime = datetime.strptime('Jan-15-2016', '%b-%d-%Y')
print "the time: ", thetime
forecast = np.array([[to_millis(thetime), 6]])
print "forecast shape: ", forecast.shape[1]
pred = clf.predict(forecast)
print "prediction: ", pred
acc = clf.score(Xt,yt)
print "prediction for ", thetime.date(), " is ", pred, " calls with ", acc * 100, "% accuracy"
