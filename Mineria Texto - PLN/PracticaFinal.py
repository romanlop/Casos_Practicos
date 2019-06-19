#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:17:44 2019

@author: Ruman
"""

import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import cess_esp as cess
from nltk.tag import UnigramTagger, BigramTagger, DefaultTagger
from sklearn.externals import joblib
import os.path as path
from nltk.tag.hmm import HiddenMarkovModelTagger
import dill
from nltk import conlltags2tree, tree2conlltags
from nltk.corpus import conll2000
import re

###############################################################################

#Corpus en castellano.
cess_sents = cess.tagged_sents()

#Frases de ejemplo. 
corpus_ejemplo=['Quiero una pizza.', 
                'Quiero 2 pollos con tomate',
                'Quiero una pizza fresca.', 
                'Quiero una pescado fresco.', 
                'Quiero una pincho de tortilla.', 
                'Quiero un pollo asado',
                'Quiero un pollo asado',
                'Quiero un pincho de tortilla',
                'Quiero dos pollos.',
                'Quiero tres pizzas y tambien quiero un pollo.',
                'Quiero una tortilla.',
                'Tambien dos salchichas.',
                'Quiero un pincho de tortilla.',
                'Quiero un pescado, tambien quiero dos cervezas.',
                'Quiero tres pollos, tambien  dos pasteles.',
                'Me gustaria comer cuatro helados, una paella ademas carne.',
                'Queria una hamburguesa.',
                'Me gustaria encargar dos paellas.',
                'Me gustaria comer cuatro filetes con dos pasteles.',
                'Ademas quiero un helado.',
                'Me gustaria comer 4 filetes con 3 pasteles.', 
                'Ademas quiero 5 helados.',
                'Quiero 1 pizza.',
                'Quiero 2 pollos con tomate',
                'Quiero 2 pollos.']


#Tiene que ser la ruta completa
corpus_file='/Users/Ruman/Desktop/DOC Master/Repositorio GITHUB/Casos_Practicos/Mineria Texto - PLN/corpus/corpus_regex.txt'

#Corpus que vamos a crear con REXPARSER.
corpus_comida=""

#Diccionario
dict_comida={}
###############################################################################


#Defición de funciones.
def realizar_pedido(option,tagger):
    pedido=1000
    while pedido!= "exit":
        print("Buenos días. ¿Cuál es su pedido?. Introduzca exit para salir.")
        pedido=input()
        if pedido != "exit":
            if int(option)==1:
                test_regex(pedido,testmode=True)

            elif int(option)==2:
                print("Opción 2. Unigram tagger")
                test_unigram(pedido,tagger)

            elif int(option)==3:
                print("Opción 3. Bigram Tagger")


            elif int(option)==4:
                print("Opción 4. NaiveBayes Classifier")
        else:
            break

    return pedido   


#Crea el fichero de corpus para luego entrenar el resto de métodos.
def train_regex(corpus_training):
    for corpus in corpus_training:
        tags = test_regex(corpus,testmode=False)
        #Guardar esto en formato IOB para entrenar los otros corpus...
        #TODO -> Cambiar este métido por apartado 4.2 -> TREES NLTK BOOK
        with open(corpus_file, 'a+', encoding="iso-8859-1") as f: 
            for item in tags:
                for l in item:
                    f.write(str(l))
                    f.write(" ")
                f.write("\n")
            f.write("\n")
            f.write("\n")
        f.close()
    return 0
    

def test_regex(frases, testmode):
    #Separamos en frases.
    frases = nltk.sent_tokenize(frases)
    #Tokenizamos.
    tokens = [nltk.word_tokenize(frase) for frase in frases]
    #Aplicamos el hidden tager 
    tagged = [hmm_tagger.tag(token) for token in tokens]
    #Comida: Detecta nombres de comida simples. Nombres seguidos de un adjetivo (pollo asado). Detecta comida tipo "pincho de tortilla" o "pollo con tomate"
    #Cantidad: Detecta letras y números
    cp = nltk.RegexpParser('''
                           COMIDA: {(<ncms000>|<ncmp000>|<ncfs000>|<Fpt>)+(<aq0ms0|aq0fs0>)*<sps00>+(<ncms000>|<ncmp000>|<ncfs000>|<da0fs0>|<Fpt>)+}   
                           COMIDA: {(<ncms000>|<ncmp000>|<ncfs000>|<Fpt>)+(<aq0ms0|aq0fs0>)*}   
                           CANTIDAD: {(<di0ms0>|<dn0cp0>|<pi0ms000>|<di0fs0>|<Z>)+}
                           ''')
    #Aplicamos Regexparses sobre nuestros tokens tageados.
    for s in tagged:
        result=cp.parse(s)
        #result.draw()
        if testmode==True:
            diccionario=diccionario_regex(result)
            print(diccionario)
        iob_tags = tree2conlltags(result)
        
    return iob_tags

#Incluye los resultados del parser regex en un diccionario que devuelve como resultado.
def diccionario_regex(t):
    cantidad=""
    for subtree in t.subtrees():
        if subtree.label() == 'CANTIDAD':
            cantidad=str(subtree.leaves()).split(" ")[0]
            cantidad=cantidad[3:-2]
            continue
        elif subtree.label() == 'COMIDA': 
            comida=str(subtree.leaves()).split(" ")[0]
            comida=comida[3:-2]
            if cantidad==str(""):
                cantidad="una"
            dict_comida[str(comida)]=str(cantidad)
            comida=""
            cantidad=""
            continue
    return dict_comida


#train UnigramTagger.
def train_unigram(fichero):
    corpus_comida=conll2000.chunked_sents(fichero, chunk_types=['COMIDA','CANTIDAD'])
    print(corpus_comida)
    train_data = [[(w,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in corpus_comida]
    print(train_data)
    tagger=nltk.UnigramTagger(train_data)
    return tagger

def test_unigram(frases, tagger):
    #Separamos en frases.
    frases = nltk.sent_tokenize(frases)
    #Tokenizamos.
    tokens = [nltk.word_tokenize(frase) for frase in frases]
    #for token in tokens:
    tagged=tagger.tag_sents(tokens)
    tree=nltk.chunk.conlltags2tree(tagged)
    return tagged
    
    

def procesado_bigram(texto_entrada):
    return 0


def procesado_naive(texto_entrada):
    return 0


##############################################################################

#Entrenamiento de los tagger
if path.exists('spanish_hmm.plk'):
    hmm_tagger = joblib.load('spanish_hmm.plk')
else:
    #Entrenamos el Hidden tagger y lo guardamos en un fichero para sucesivas ocasiones
    hmm_tagger = HiddenMarkovModelTagger.train(cess_sents)
    with open('spanish_hmm.plk', 'wb') as pickle_file:
        dill.dump(hmm_tagger, pickle_file) 


#CAMBIAR ESTO -> PRIMERO EJECUTAR EL REGEX LUEGO EL RESTO EN ORDEN...
#Menú principal
print("Selección una Opción:")
print("1.Entrenamiento RegexParser.")
print("2.Test.")
print("3.Salir.")
opcion=input()

if int(opcion)==1:
    print("Entrenando RegexParser...")
    train_regex(corpus_ejemplo)
        

elif int(opcion)==2:
    print("####################################")
    print("1. Test RegexParser")
    print("2. Test Unigram tagger")
    print("3. Test Bigram Tagger")
    print("4. Test NaiveBayes Classifier")
    print("Exit para salir.")
    opcion3=input()
    if int(opcion3)==1:
        print("1. Test RegexParser")
        pedido=realizar_pedido(1,False)

    elif int(opcion3)==2:
        print("Opción 2. Test Unigram tagger")
        unigram_tagger=train_unigram(corpus_file)
        pedido=realizar_pedido(2,unigram_tagger)
        
    elif int(opcion3)==3:
        print("Opción 3. Test Bigram Tagger")
        pedido=realizar_pedido()
        
    elif int(opcion3)==4:
        print("Opción 4. Test NaiveBayes Classifier")
        pedido=realizar_pedido()
        
    else:
        print("Adios") 

elif int(opcion)==3:
    print("Adios")   
    

else:
    print("Adios") 



