#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:31:23 2019

@author: Ruman
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

#Carga de los datos
datos = pd.read_csv("data/crime_data.csv",sep=",",keep_default_na=False) 

#Vamos a centrarnos inicialmente en definir el número de clústers en los que segmentar nuestros datos.
def clasificacion_silhouette (nombre, n_clusters, data):
    silhouette = []
    #Para silhouette hay que partir de 2 clusters. En caso contrario casca. No tiene sentido aplicarlo.
    for i in range(2,n_clusters+1):
        kmeans = KMeans(n_clusters = i,
                        random_state = 1).fit(data)
        labels = kmeans.labels_
        silhouette.append(metrics.silhouette_score(data, labels))
        
    plt.plot(range(2,n_clusters+1), silhouette, 'ro-')   
    plt.title("Silhouette Coeficcient: " + nombre)
    plt.xlabel("Número Clústers")
    plt.ylabel("silhouette")
    plt.show()
    
    
def clasificacion_codo (nombre, n_clusters, data):
    inertia = []
    for i in range(1,n_clusters+1):
        kmeans = KMeans(n_clusters = i,
                        random_state = 1).fit(data)
        inertia.append(kmeans.inertia_)
        
    plt.plot(range(1,n_clusters+1), inertia, 'ro-')   
    plt.title("Dispersión: " + nombre)
    plt.xlabel("Número Clústers")
    plt.ylabel("Inertia / Dispersión")    
    plt.show()
    

def clasificacion_elbow  (nombre, max_value, data):
    distortions = []
    K = range(1,max_value)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title("Elbow: " + nombre)
    plt.show()
    
"""
El PCA se puede utilizar, entre otros, para la representación de conjuntos de datos que 
tengan más de dos dimensiones. Para ello tenemos que fijar n=2.
"""   

pca = PCA(n_components=2)
pca.fit(datos.loc[:, 'Murder'::])
#Transformamos los datos en base a los nuevos componentes
componentes_principales = pca.fit_transform(datos.loc[:, 'Murder'::])
#Pintamos la gráfica
plt.title("Datos sobre componentes principales.")
plt.scatter(*zip(*componentes_principales))
plt.show()

#Aparentemente no es fácil determinar el número de clústeres. 
#Vamos a testear silhoute hasta el número de estados
clasificacion_silhouette("Estados USA - Crimenes", 20, datos.loc[:, 'Murder'::])
#Vemos que a medida que avanzamos el valor de silhoutte decrece.

#Vamos a testear método codo hasta el número de estados
clasificacion_codo("Estados USA - Crimenes", 10, datos.loc[:, 'Murder'::])
#Vemos que a medida que avanzamos el valor de silhoutte decrece.

clasificacion_elbow("Estados USA - Crimenes", 25, datos.loc[:, 'Murder'::])
#Se aprecia un codo en el valor 5!!!!!!


#Vamos a dar dos soluciones. Una con 2 clústeres y otra con 5. 

kmeans = KMeans(n_clusters = 2,
                random_state = 1).fit(componentes_principales)

datos['classes'] = kmeans.predict(componentes_principales)

for i in list(set(datos.classes)):
    print ("Estados integrantes del Clúster número ",i,":")
    print (datos.loc[datos['classes'] == i]['State'])


plt.title("Datos sobre componentes principales.")
plt.scatter(*zip(*componentes_principales))
plt.scatter(kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker = '*',
        s = 250,
        color = 'red')
plt.show()


kmeans = KMeans(n_clusters = 5,
                random_state = 1).fit(componentes_principales)

datos['classes'] = kmeans.predict(componentes_principales)

for i in list(set(datos.classes)):
    print ("Estados integrantes del Clúster número ",i,":")
    print (datos.loc[datos['classes'] == i]['State'])

print(kmeans.cluster_centers_[:, 0])

plt.title("Datos sobre componentes principales.")
plt.scatter(*zip(*componentes_principales))
plt.scatter(kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker = '*',
        s = 250,
        color = 'red')
plt.show()



