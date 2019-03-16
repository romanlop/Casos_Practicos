#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:09:10 2019

@author: Ruman
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
import sklearn.metrics as mtr
import numpy as np




def calculateVIF(data):
    features = list(data.columns)
    num_features = len(features)
    
    #vamos a buscar el R^2 para poder aplicar la formula de VIF
    model = LinearRegression()
    
    #Crea un dataframe con las columnas de los datos originales que pasamos como parámetro. Crea una fila con NAN
    result = pd.DataFrame(index = ['VIF'], columns = features)
    #Sustituimos los NAN por ceros.
    result = result.fillna(0)
    
    #Iteramos las características
    for ite in range(num_features):
        #esta variable almacena un listado con las caracteristicas
        x_features = features[:]
        #en esta almacenamos la caracteristica sobre la que estamos iterando. Una en cada iteración del bucle.
        y_feature = features[ite]
        #eliminamos dicha variable del conjunto de variables.
        x_features.remove(y_feature)
        
        x = data[x_features]
        #en y guardamos la carateristica que hemos seleccionado en este paso del bucle, la trataremos como target en la regresión.
        y = data[y_feature]
        
        model.fit(data[x_features], data[y_feature])
        
        #Aplicamos la fórmula de VIF
        result[y_feature] = 1/(1 - model.score(data[x_features], data[y_feature]))
    
    return result


def selectDataUsingVIF(data, max_VIF = 5):
    result = data.copy(deep = True)
    
    VIF = calculateVIF(result)
    
    while VIF.as_matrix().max() > max_VIF:
        col_max = np.where(VIF == VIF.as_matrix().max())[1][0]
        features = list(result.columns)
        features.remove(features[col_max])
        result = result[features]
        
        VIF = calculateVIF(result)
        
    return result
    


#Carga de los datos
datos = pd.read_csv("data/auto.csv",sep=",",keep_default_na=False) 

#Realizar limpieza nulos en caso de existir. 
datos.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

#Separamos los datos entre variables independientes y variable dependediente (consumo)
consumo = datos['mpg']
datos = datos.drop('mpg',axis=1)


#Dividir los datos para train y test.
train_car, test_car, train_mpg, test_mpg = train_test_split( datos, 
                                                            consumo, 
                                                            test_size=1/7.0, 
                                                            random_state=0)


print("\n########################################################")
print("Modelo Lineal:")
#Definimos y entrenamos el modelo
model = LinearRegression().fit(train_car, train_mpg)
#Vamos a ver las medidas que obtenemos sobre el conjunto de train.
print('Métricas conjunto de Train:')
mpg_predict = model.predict(train_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict, train_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict, train_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict, train_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict, train_mpg))

#Vamos a ver las medidas que obtenemos sobre el conjunto de test.
print('Métricas conjunto de Test:')
mpg_predict_test = model.predict(test_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict_test, test_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict_test, test_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict_test, test_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict_test, test_mpg))


#Vamos a probar sin definir el Intercept
print("\n########################################################")
print("Modelo Lineal sin Intercept. Se obtienen peores resultados")
#Definimos y entrenamos el modelo
model = LinearRegression(fit_intercept=False).fit(train_car, train_mpg)

#Vamos a ver las medidas que obtenemos sobre el conjunto de train.
print('Métricas conjunto de Train:')
mpg_predict = model.predict(train_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict, train_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict, train_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict, train_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict, train_mpg))

#Vamos a ver las medidas que obtenemos sobre el conjunto de test.
print('Métricas conjunto de Test:')
mpg_predict_test = model.predict(test_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict_test, test_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict_test, test_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict_test, test_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict_test, test_mpg))


#Sospechamos que algunas variables realmente no aportan al modelo. 
print("\n########################################################")
print("Modelo Lasso. No mejoramos el modelo")

model = Lasso(alpha=0.05, normalize=False).fit(train_car, train_mpg)
print("Vemos que características ha utilizado para el modelo:", model.coef_)
print("El intercept es:", model.intercept_) 

mpg_predict = model.predict(train_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict, train_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict, train_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict, train_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict, train_mpg))

#Vamos a ver las medidas que obtenemos sobre el conjunto de test.
print('Métricas conjunto de Test:')
mpg_predict_test = model.predict(test_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict_test, test_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict_test, test_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict_test, test_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict_test, test_mpg))


#Sospechamos que algunas variables realmente no aportan al modelo. 
print("\n########################################################")
print("Modelo Ridge. No mejoramos el modelo")

model = Ridge(alpha=1.5).fit(train_car, train_mpg)
print("Vemos que características ha utilizado para el modelo:", model.coef_)
print("El intercept es:", model.intercept_) 

mpg_predict = model.predict(train_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict, train_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict, train_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict, train_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict, train_mpg))

#Vamos a ver las medidas que obtenemos sobre el conjunto de test.
print('Métricas conjunto de Test:')
mpg_predict_test = model.predict(test_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict_test, test_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict_test, test_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict_test, test_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict_test, test_mpg))



#Eliminamos variables independientes para evitar multicolinealidad y volvemos a probar con un modelo Lineal.
datos = selectDataUsingVIF(datos.copy(deep = True))  

#Dividir los datos para train y test.
train_car, test_car, train_mpg, test_mpg = train_test_split( datos, 
                                                            consumo, 
                                                            test_size=1/7.0, 
                                                            random_state=0)

print("\n########################################################")
print("Modelo Lineal:")
#Definimos y entrenamos el modelo
model = LinearRegression().fit(train_car, train_mpg)
#Vamos a ver las medidas que obtenemos sobre el conjunto de train.
print('Métricas conjunto de Train:')
mpg_predict = model.predict(train_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict, train_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict, train_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict, train_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict, train_mpg))

#Vamos a ver las medidas que obtenemos sobre el conjunto de test.
print('Métricas conjunto de Test:')
mpg_predict_test = model.predict(test_car)
print('Error cuadrático medio', mtr.mean_squared_error(mpg_predict_test, test_mpg))
print('Error absoluto medio', mtr.mean_absolute_error(mpg_predict_test, test_mpg))
print('Mediana del error absoluto', mtr.median_absolute_error(mpg_predict_test, test_mpg))
print('R2 en entrenamiento es: ', r2_score(mpg_predict_test, test_mpg))