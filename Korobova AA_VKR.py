#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import openpyxl
import seaborn
import matplotlib.pyplot as plt
from matplotlib import figure
import sys
import pylab as pl
from pylab import figure


# In[4]:


x_bp_df = pd.read_excel('C:/Users/Alexandra/Desktop/вкр/X_bp.xlsx', sheet_name='X_bp.csv')
x_nup_df = pd.read_excel('C:/Users/Alexandra/Desktop/вкр/X_nup.xlsx', sheet_name='X_nup.csv')


# In[5]:


npbp_df = x_bp_df.merge(x_nup_df, left_index=True, right_index=True, how='inner')


# In[6]:


npbp_df.drop(columns =['Unnamed: 0_x', 'Unnamed: 0_y'],axis=1,inplace=True)


# In[7]:


npbp_df.describe()


# In[8]:


npbp_df = npbp_df.dropna()


# In[9]:


npbp_df.describe()


# In[10]:


for col in npbp_df.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = npbp_df[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/Гистограмма.pdf')
    plt.show


# In[11]:


seaborn.pairplot(npbp_df, height=2.5)


# In[128]:


plt.figure(figsize = (10,3))
seaborn.heatmap(npbp_df.corr(), cmap= 'rainbow', annot = True, linewidths=1, linecolor='black' )


# In[13]:


npbp_df.mean()


# In[14]:


npbp_df.median()


# In[15]:


npbp_df.describe().transpose()[['mean','std']]


# In[16]:


npbp_df.boxplot(rot=90)


# In[17]:


for x in ['Соотношение матрица-наполнитель']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[18]:


for x in ['Плотность, кг/м3']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[19]:


for x in ['модуль упругости, ГПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[20]:


for x in ['Количество отвердителя, м.%']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[21]:


for x in ['Содержание эпоксидных групп,%_2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[22]:


for x in ['Температура вспышки, С_2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[23]:


for x in ['Поверхностная плотность, г/м2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[24]:


for x in ['Модуль упругости при растяжении, ГПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[25]:


for x in ['Прочность при растяжении, МПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[26]:


for x in ['Потребление смолы, г/м2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[27]:


for x in ['Угол нашивки, град']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[28]:


for x in ['Шаг нашивки']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[29]:


for x in ['Плотность нашивки']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan


# In[30]:


npbp_df.isnull().sum()


# In[31]:


npbp_df = npbp_df.dropna(axis = 0)


# In[32]:


npbp_df.isnull().sum()


# In[33]:


npbp_df.duplicated().sum()


# In[34]:


npbp_df.info()


# In[35]:


npbp_df.describe()


# In[36]:


for col in npbp_df.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = npbp_df[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/Гистограмма.pdf')
    plt.show
    


# In[37]:


seaborn.pairplot(npbp_df, height=2.5)


# In[38]:


plt.figure(figsize = (14,6))
seaborn.heatmap(npbp_df.corr(), cmap= 'rainbow', annot = True, linewidths=1, linecolor='black' )


# In[39]:


npbp_df.mean()


# In[40]:


npbp_df.median()


# In[41]:


npbp_df.describe().transpose()[['mean','std']]


# In[42]:


from sklearn.preprocessing import MinMaxScaler


# In[43]:


minmax_scaler = MinMaxScaler ()
df_norm = minmax_scaler.fit_transform(np.array(npbp_df[['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']]))


# In[44]:


df_norm = pd.DataFrame(data = df_norm, columns = ['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки'])
df_norm.head()


# In[45]:


df_norm.describe()


# In[46]:


for col in df_norm.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = df_norm[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/ГистограммаНорм.pdf')
    plt.show


# In[47]:


boxplot = df_norm.boxplot(rot=90)


# In[48]:


import splitfolders 
from sklearn.model_selection import train_test_split


# In[49]:


x_upr=df_norm.drop(['Модуль упругости при растяжении, ГПа'], axis=1)
x_pr=df_norm.drop(['Прочность при растяжении, МПа'], axis=1)
y_upr=df_norm['Модуль упругости при растяжении, ГПа']
y_pr=df_norm['Прочность при растяжении, МПа']

x_train_upr, x_test_upr, y_train_upr, y_test_upr=train_test_split(x_upr, y_upr, test_size=0.3, random_state=1)
x_train_pr, x_test_pr, y_train_pr, y_test_pr=train_test_split(x_pr, y_pr, test_size=0.3, random_state=1)


# In[50]:


print('Размер тренировочного датасета на входе:', x_train_upr.shape)
print('Размер тестового датасета на входе:', x_test_upr.shape)
print('Размер тренировочного датасета на выходе:', y_train_upr.shape)
print('Размер тестового датасета на выходе:', y_test_upr.shape)


# In[51]:


print('Размер тренировочного датасета на входе:', x_train_pr.shape)
print('Размер тестового датасета на входе:', x_test_pr.shape)
print('Размер тренировочного датасета на выходе:', y_train_pr.shape)
print('Размер тестового датасета на выходе:', y_test_pr.shape)


# In[52]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


# In[53]:


Ir = LinearRegression()
Ir_params = { 'fit_intercept' : ['True','False']}
GSCV_ir_upr = GridSearchCV(Ir, Ir_params, n_jobs=-1, cv=10)
GSCV_ir_upr.fit(x_train_upr, y_train_upr)
GSCV_ir_upr.best_params_
{'fit_intercept': 'True'}


# In[54]:


Ir_upr = GSCV_ir_upr.best_estimator_
print(f'R2-score LR для модуля упругости при растяжении: {Ir_upr.score(x_test_upr, y_test_upr).round(3)}')


# In[55]:


Ir_upr_result = pd.DataFrame({
    'Model': 'LinearRegression_upr',
    'MAE': mean_absolute_error(y_test_upr, Ir_upr.predict(x_test_upr)),
    'R2 score':Ir_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])


# In[56]:


Ir_upr_result


# In[57]:


from sklearn.neighbors import KNeighborsRegressor


# In[58]:


knr = KNeighborsRegressor()
knr_params = {'n_neighbors': range(1,301,5),
             'weights': ['uniform','distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
GSCV_knr_upr = GridSearchCV(knr, knr_params, n_jobs=-1, cv=10, error_score='raise')
GSCV_knr_upr.fit(x_train_upr, y_train_upr)
GSCV_knr_upr.best_params_


# In[59]:


knr_upr = GSCV_knr_upr.best_estimator_
print(f'R2-score KNR для модуля упругости при растяжении: {knr_upr.score(x_test_upr, y_test_upr).round(3)}')


# In[60]:


knr_upr_result = pd.DataFrame({
    'Model': 'KNeighborsRegressor_upr',
    'MAE': mean_absolute_error(y_test_upr, knr_upr.predict(x_test_upr)),
    'R2 score':knr_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])


# In[61]:


knr_upr_result


# In[62]:


from sklearn.tree import DecisionTreeRegressor


# In[63]:


dt = DecisionTreeRegressor()
dt_params = { 'criterion': ['mse', 'friedman_mse', 'mae']}
GSCV_dt_upr = GridSearchCV(dt, dt_params, n_jobs=-1, cv=10)
GSCV_dt_upr.fit(x_train_upr, y_train_upr)
GSCV_dt_upr.best_params_


# In[64]:


dt_upr = GSCV_dt_upr.best_estimator_
print(f'R2-score KNR для модуля упругости при растяжении: {dt_upr.score(x_test_upr, y_test_upr).round(3)}')


# In[65]:


dt_upr_result = pd.DataFrame({
    'Model': 'DecisionTreeRegressor',
    'MAE': mean_absolute_error(y_test_upr, dt_upr.predict(x_test_upr)),
    'R2 score':dt_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])


# In[66]:


dt_upr_result


# In[67]:


Ir = LinearRegression()
Ir.fit(x_train_pr, y_train_pr)
y_pred_ir = Ir.predict(x_test_pr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_ir, 'g', label='prediction')
pl.plot(y_test_pr.values, label='actual')
pl.grid(True);


# In[68]:


Ir = LinearRegression()
Ir.fit(x_train_upr, y_train_upr)
y_pred_ir = Ir.predict(x_test_upr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_ir, 'g', label='prediction')
pl.plot(y_test_upr.values, label='actual')
pl.grid(True);


# In[70]:


param_grid = {'n_neighbors': range(1,50)}
gs = GridSearchCV(knr, param_grid, cv=10, verbose = 1, n_jobs=-1)
gs.fit(x_train_pr,y_train_pr)
knn_3 = gs.best_estimator_
gs.best_params_


# In[71]:


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train_pr, y_train_pr)
y_pred_knn = knn.predict(x_test_pr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_knn, 'g', label='prediction')
pl.plot(y_test_pr.values, label='actual')
pl.grid(True);


# In[72]:


param_grid = {'n_neighbors': range(1,50)}
gs = GridSearchCV(knn, param_grid, cv=10, verbose = 1, n_jobs=-1)
gs.fit(x_train_upr,y_train_upr)
knn_3 = gs.best_estimator_
gs.best_params_


# In[73]:


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train_upr, y_train_upr)
y_pred_knn = knn.predict(x_test_upr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_knn, 'g', label='prediction')
pl.plot(y_test_upr.values, label='actual')
pl.grid(True);


# In[74]:


param_grid = {'criterion': ['friedman_mse']}
gs = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10, verbose = 1, n_jobs=-1)
gs.fit(x_train_pr,y_train_pr)
dt_3 = gs.best_estimator_
gs.best_params_


# In[75]:


dt = DecisionTreeRegressor()
dt.fit(x_train_pr, y_train_pr)
y_pred_dt = dt.predict(x_test_pr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_dt, 'g', label='prediction')
pl.plot(y_test_pr.values, label='actual')
pl.grid(True);


# In[76]:


param_grid = {'criterion': ['friedman_mse']}
gs = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10, verbose = 1, n_jobs=-1)
gs.fit(x_train_upr,y_train_upr)
dt_3 = gs.best_estimator_
gs.best_params_


# In[77]:


dt = DecisionTreeRegressor()
dt.fit(x_train_upr, y_train_upr)
y_pred_dt = dt.predict(x_test_upr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_dt, 'g', label='prediction')
pl.plot(y_test_upr.values, label='actual')
pl.grid(True);


# In[78]:


input_columns_names = ['Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']
output_columns_names = ['Соотношение матрица-наполнитель']
x = df_norm[input_columns_names]
y = df_norm[output_columns_names]


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[80]:


knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [1,2,5,10,20]}
GSCV = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, verbose=2)
GSCV.fit(x_train, y_train)
GSCV.best_params_


# In[81]:


knn.fit(x_train, y_train)
prediction=knn.predict(x_test)
np.mean((y_test - prediction)*(y_test - prediction))


# In[82]:


input_columns_names = ['Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']
output_columns_names = ['Соотношение матрица-наполнитель']
x = df_norm[input_columns_names]
y = df_norm[output_columns_names]


# In[83]:


print(x.shape, y.shape)


# In[84]:


seaborn.pairplot(pd.DataFrame(np.column_stack([x, y])), diag_kind='kde')


# In[85]:


pd.DataFrame(np.column_stack(([x, y])))


# In[86]:


from keras.models import Sequential
from keras import models


# In[87]:


def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model


# In[88]:


model = get_model(12,1)
model.summary()


# In[89]:


hist = model.fit(x, y, verbose=0, epochs=2000, validation_data = (x_train, y_train))


# In[90]:


df = pd.DataFrame(hist.history)


# In[91]:


import matplotlib.pyplot as plt


# In[92]:


plt.plot(df)


# In[131]:


score = model.evaluate(x_test, y_test, verbose=1)
print('Потери на тесте:', score[0])
print('Точность на тесте:', score[1])


# In[94]:


x


# In[95]:


prediction = model.predict(x)


# In[96]:


prediction


# In[97]:


np.mean(np.abs((y-prediction)), axis=0)


# In[98]:


np.abs(y-prediction)


# In[99]:


np.mean((y-prediction)*(y-prediction), axis=0)


# In[100]:


model_path = 'C:/Users/Alexandra/Desktop/вкр/models/my_model_2'


# In[101]:


model.save(model_path)

