#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:10:17 2018

@author: Ruman

Dado un archivo que contiene en cada línea una palabra o conjunto de palabras 
seguido de un valor numérico, denominado “sentimiento”, y un conjunto de tweets, 
se pide calcular para cada tweet un valor denominado “sentimiento del tweet”, 
que se obtiene como la suma de los “sentimientos” de los términos que aparecen en el tweet.
"""

import re
import json

"""
#elimina signos de puntuacion.
def remove_punctuation ( text ):
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)"""

#devuelve el contenido del fichero de sentimientos
def lectura_sentimientos(fichero):
    valores = {}
    with open(fichero) as s: 
        for linea in s:
            termino,valor = linea.split("\t")
            valores[termino] = valor.rstrip('\n')
            #Controlamos la posibilidad de que en una fila exista mas de un termino
            #for t in termino.split():
            #    valores[t] = valor.rstrip('\n')
    return valores

#procesa el fichero que contiene los json
def lectura_twitter_json(fichero_json):
    tweets = []
    with open(fichero_json) as t: 
        for linea in t:
            tweets.append(linea.rstrip('\n'))
    return tweets


#a partir de una lista conteniendo todos los JSON, extrae el tweet y los mete en otra lista.
def tweet_parser(lista_tweets_json):
    tweets = []
    for lt in lista_tweets_json:
        json_datos=json.loads(lt)
        if 'created_at' in json_datos:
            tweets.append(json_datos['text'].rstrip('\n'))
    return tweets



#calculo del valor del sentimiento
#para cada TWEET, los paso a una lista.
#Itero el diccionario y busco sobre la lista anterior. 
#Imprimo el calculo y voy al siguiente tweet            
def calcular_sentimiento(sent,twt):
    if len(twt) == 0: 
        print("El fichero de Tweets está vacío.")
    else:
        for t in twt:
            calc_sentimiento=0
            #print("\nTerminos:",terminos,)
            #recorro el diccionario y chequeo para cada clave si está en la lista de elementos del tweet
            for s in sent:
                patron = re.compile(r'\b{}\b'.format(s))
                if patron.search(t)!= None:
                    #tenemos en cuenta el número de ocurrencias de cada valor del diccionario
                    calc_sentimiento=calc_sentimiento+(int(sent[s])*t.count(s))
            print ("*************************************************+****************")
            print ("El siguiente TWEET:",t,"Tiene un sentimiento asociado de:",calc_sentimiento)



#Programa Principal     
#"conf/sentimientos.txt"
#"tweets/tweet1.txt"
""" PROGRAMA FINAL
print("Ayuda. El fichero de Tweets puede contener varios tweet separados por retorno de carro o un tweet individual.")
sentimientos_file=input("Introduzca la ruta al fichero de sentimientos:")
tweet_file=input("Introduzca la ruta al fichero que contiene los Tweet:")   
        
sentimientos=lectura_sentimientos(sentimientos_file)
tweet=lectura_twitter(tweet_file)
calcular_sentimiento(sentimientos,tweet)"""

sentimientos=lectura_sentimientos("conf/Sentimientos.txt")
lista_json=lectura_twitter_json("tweets/salida_tweets.json")
lista_tweets=tweet_parser(lista_json)
calcular_sentimiento(sentimientos,lista_tweets)
