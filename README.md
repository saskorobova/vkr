# vkr 
# Подготовка данных
Первым делом производится импорт нужных библиотек
```
import pandas as pd
import numpy as np
import openpyxl
import seaborn
import matplotlib.pyplot as plt
from matplotlib import figure
import sys
import pylab as pl
from pylab import figure
```
Файл формата exel, но для дальнейший работы нужно прочитать его в формате csv
```
x_bp_df = pd.read_excel('C:/Users/Alexandra/Desktop/вкр/X_bp.xlsx', sheet_name='X_bp.csv')
x_nup_df = pd.read_excel('C:/Users/Alexandra/Desktop/вкр/X_nup.xlsx', sheet_name='X_nup.csv')
```
Произведено объединение по индексу тип объединения INNER
```
npbp_df = x_bp_df.merge(x_nup_df, left_index=True, right_index=True, how='inner')
```
Отсечение неинформативных колонок
```
npbp_df.drop(columns =['Unnamed: 0_x', 'Unnamed: 0_y'],axis=1,inplace=True)
```

Описателььная статистика первоначального набора

```
npbp_df.describe()
```

Построение гистограмм первоначальных данных
```
for col in npbp_df.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = npbp_df[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/Гистограмма.pdf')
    plt.show
```
Графики взаимосвязей первоначального набора данных
```
seaborn.pairplot(npbp_df, height=2.5)
```
Графики корреляции первоначального набора данных
```
plt.figure(figsize = (10,3))
seaborn.heatmap(npbp_df.corr(), cmap= 'rainbow', annot = True, linewidths=1, linecolor='black' )
```
Статистические параметры
```
npbp_df.mean()
npbp_df.median()
npbp_df.describe().transpose()[['mean','std']]
```
Ящик с усами (поиск выбросов)
```
npbp_df.boxplot(rot=90)
```
Выявление выбросов с помощью метода квартилей и удадение
```
for x in ['Соотношение матрица-наполнитель']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Плотность, кг/м3']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['модуль упругости, ГПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Количество отвердителя, м.%']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Содержание эпоксидных групп,%_2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Температура вспышки, С_2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Поверхностная плотность, г/м2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Модуль упругости при растяжении, ГПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Прочность при растяжении, МПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Потребление смолы, г/м2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Угол нашивки, град']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Шаг нашивки']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Плотность нашивки']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
npbp_df.isnull().sum()

npbp_df = npbp_df.dropna(axis = 0)

npbp_df.isnull().sum()

npbp_df.duplicated().sum()
```
Статистика, гистограммы и графики после удаления выбросов. После удаления выбросов ничего, что могло бы нас заинтересовать, не проявилось
```
npbp_df.info()
npbp_df.describe()
for col in npbp_df.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = npbp_df[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/Гистограмма.pdf')
    plt.show
    
    seaborn.pairplot(npbp_df, height=2.5)
    
    plt.figure(figsize = (14,6))
seaborn.heatmap(npbp_df.corr(), cmap= 'rainbow', annot = True, linewidths=1, linecolor='black' )

npbp_df.mean()

npbp_df.median()

npbp_df.describe().transpose()[['mean','std']]

```
Нормализация
```
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler ()
df_norm = minmax_scaler.fit_transform(np.array(npbp_df[['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']]))
df_norm = pd.DataFrame(data = df_norm, columns = ['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки'])
```
Статистика и гистраграммы после нормализации
```
df_norm.head()
df_norm.describe()
for col in df_norm.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = df_norm[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/ГистограммаНорм.pdf')
    plt.show
boxplot = df_norm.boxplot(rot=90)
```
Разметка данных
```
import splitfolders 
from sklearn.model_selection import train_test_split
x_upr=df_norm.drop(['Модуль упругости при растяжении, ГПа'], axis=1)
x_pr=df_norm.drop(['Прочность при растяжении, МПа'], axis=1)
y_upr=df_norm['Модуль упругости при растяжении, ГПа']
y_pr=df_norm['Прочность при растяжении, МПа']

x_train_upr, x_test_upr, y_train_upr, y_test_upr=train_test_split(x_upr, y_upr, test_size=0.3, random_state=1)
x_train_pr, x_test_pr, y_train_pr, y_test_pr=train_test_split(x_pr, y_pr, test_size=0.3, random_state=1)
print('Размер тренировочного датасета на входе:', x_train_upr.shape)
print('Размер тестового датасета на входе:', x_test_upr.shape)
print('Размер тренировочного датасета на выходе:', y_train_upr.shape)
print('Размер тестового датасета на выходе:', y_test_upr.shape)

print('Размер тренировочного датасета на входе:', x_train_pr.shape)
print('Размер тестового датасета на входе:', x_test_pr.shape)
print('Размер тренировочного датасета на выходе:', y_train_pr.shape)
print('Размер тестового датасета на выходе:', y_test_pr.shape)
```
# Решение задачи с помощью метода регрессии
Импорт нужных библиотек
```
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
```
Поиск детерменированного коэфициэнта линейной регрессии
```
Ir = LinearRegression()
Ir_params = { 'fit_intercept' : ['True','False']}
GSCV_ir_upr = GridSearchCV(Ir, Ir_params, n_jobs=-1, cv=10)
GSCV_ir_upr.fit(x_train_upr, y_train_upr)
GSCV_ir_upr.best_params_
{'fit_intercept': 'True'}
Ir_upr = GSCV_ir_upr.best_estimator_
print(f'R2-score LR для модуля упругости при растяжении: {Ir_upr.score(x_test_upr, y_test_upr).round(3)}')

Ir_upr_result = pd.DataFrame({
    'Model': 'LinearRegression_upr',
    'MAE': mean_absolute_error(y_test_upr, Ir_upr.predict(x_test_upr)),
    'R2 score':Ir_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])

Ir_upr_result
```
Поиск детерменированного коэфициэнта регрессии с помощью метода наиближайших соседей
```
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr_params = {'n_neighbors': range(1,301,5),
             'weights': ['uniform','distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
GSCV_knr_upr = GridSearchCV(knr, knr_params, n_jobs=-1, cv=10, error_score='raise')
GSCV_knr_upr.fit(x_train_upr, y_train_upr)
GSCV_knr_upr.best_params_

knr_upr = GSCV_knr_upr.best_estimator_
print(f'R2-score KNR для модуля упругости при растяжении: {knr_upr.score(x_test_upr, y_test_upr).round(3)}')

knr_upr_result = pd.DataFrame({
    'Model': 'KNeighborsRegressor_upr',
    'MAE': mean_absolute_error(y_test_upr, knr_upr.predict(x_test_upr)),
    'R2 score':knr_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])

knr_upr_result
```
Поиск детерменированного коэфициэнта регрессии дерева
```
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt_params = { 'criterion': ['mse', 'friedman_mse', 'mae']}
GSCV_dt_upr = GridSearchCV(dt, dt_params, n_jobs=-1, cv=10)
GSCV_dt_upr.fit(x_train_upr, y_train_upr)
GSCV_dt_upr.best_params_

dt_upr = GSCV_dt_upr.best_estimator_
print(f'R2-score KNR для модуля упругости при растяжении: {dt_upr.score(x_test_upr, y_test_upr).round(3)}')

dt_upr_result = pd.DataFrame({
    'Model': 'DecisionTreeRegressor',
    'MAE': mean_absolute_error(y_test_upr, dt_upr.predict(x_test_upr)),
    'R2 score':dt_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])

dt_upr_result
```
