#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:17:44 2019

@author: Ruman
"""

import sys
import nltk
from nltk import word_tokenize

#Defición de funciones.
def realizar_pedido(opcion):
    pedido=1000
    while pedido!= "exit":
        print("Buenos días. ¿Cuál es su pedido?. Introduzca exit para salir.")
        pedido=input()
        print("procesando...")
    #LLAMAR AL METODO CORRESPONDIENTE EN FUNCIÓN DE LA OPCIÓN SELECCIONADA.
    return pedido



#Menú principal
print("Selección del modo de funcionamiento:")
print("1. RegexParser")
print("2. Bigram tagger")
print("3. Unigram Tagger")
print("4. NaiveBayes Classifier")
option = input()

if int(option)==1:
    print("Ha seleccionado la opción 1. RegexParser")
    realizar_pedido(1)
elif int(option)==2:
    print("Opción 2. Bigram tagger")
    realizar_pedido(2)
elif int(option)==3:
    print("Opción 3. Unigram Tagger")
    realizar_pedido(3)
elif int(option)==4:
    print("Opción 4. NaiveBayes Classifier")
    realizar_pedido(4)
else:
    exit(0)
    





