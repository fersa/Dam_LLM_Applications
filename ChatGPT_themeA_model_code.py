
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_excel('ThemeA_data_fmt03.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Feature engineering
df['day'] = np.arange(len(df))
df['sin'] = np.sin(2*np.pi*df.index.dayofyear/365.25)
df['cos'] = np.cos(2*np.pi*df.index.dayofyear/365.25)

features = ['Water_Level','Ta','Tb','Rainfall','day','sin','cos']

# Cases
caseA = df['2000':'2012']
caseB = df['2013-01-01':'2013-06-30']
caseC = df['2013-07-01':'2017-12-31']

targets = ['CB2_236_196','CB3_195_161','C4_C5','PZCB2','PZCB3','Seepage']

predsB = pd.DataFrame(index=caseB.index)
predsC = pd.DataFrame(index=caseC.index)

for t in targets:
    train = caseA[features + [t]].dropna()
    X = train[features].values
    y = train[t].values

    med = GradientBoostingRegressor(loss='huber')
    med.fit(X,y)

    lower = GradientBoostingRegressor(loss='quantile', alpha=0.05)
    upper = GradientBoostingRegressor(loss='quantile', alpha=0.95)
    lower.fit(X,y)
    upper.fit(X,y)

    XB = caseB[features].values
    predsB[t] = med.predict(XB)
    predsB[t+'_low'] = lower.predict(XB)
    predsB[t+'_upp'] = upper.predict(XB)

    XC = caseC[features].values
    predsC[t] = med.predict(XC)
    predsC[t+'_low'] = lower.predict(XC)
    predsC[t+'_upp'] = upper.predict(XC)

with pd.ExcelWriter('themeA_predictions_BC.xlsx') as writer:
    predsB.to_excel(writer, sheet_name='CaseB')
    predsC.to_excel(writer, sheet_name='CaseC')
