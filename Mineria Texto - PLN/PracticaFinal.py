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

#Frases de ejemplo
#TODO -> Meterlo en una excel o csv o similar...
corpus_ejemplo=['Quiero una pizza. Quiero dos pollos.',
                'Quiero una tortilla. Tambien dos salchichas.',
                'Quiero un pescado, tambien quiero dos cervezas.',
                'Quiero tres pollos, tambien  dos pasteles.',
                'Me gustaria comer cuatro helados, una paella ademas carne.',
                'Queria una hamburguesa.',
                'Me gustaria encargar dos paellas.',
                'Me gustaria comer cuatro filetes con dos pasteles. Ademas quiero un helado.']

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
            #f.write(". . O")
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
    #TODO: MEJORARLO CON NÚMEROS Y DEMÁS... Pincho de tortilla...
    cp = nltk.RegexpParser('''
                           COMIDA: {(<ncms000>|<ncmp000>|<ncfs000>)+}
                           CANTIDAD: {(<di0ms0>|<dn0cp0>|<pi0ms000>|<di0fs0>)+}
                           ''')
    #Aplicamos Regexparses sobre nuestros tokens tageados.
    for s in tagged:
        result=cp.parse(s)
        #result.draw()
        if testmode==True:
            diccionario=diccionario_regex(result)
            print(diccionario)
        iob_tags = tree2conlltags(result)
        #print(iob_tags)
    return iob_tags

#TODO. Si no detecta ninguna cantidad casca.
def diccionario_regex(t):
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
    train_data = [[(w,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in corpus_comida]
    print(train_data)
    tagger=nltk.UnigramTagger(train_data)
    """unigramChk = UnigramChunker(corpus_comida)
    print("Entrenado!!!")
    sentence = [('Quiero', 'sps00'), ('una', 'di0fs0'), ('pizza', 'ncfs000'), ('y', 'cc'), ('dos', 'dn0cp0'), ('pollos', 'ncmp000')]"""
    return tagger

def test_unigram(frases, tagger):
    print("TEST")
    #Separamos en frases.
    frases = nltk.sent_tokenize(frases)
    #Tokenizamos.
    tokens = [nltk.word_tokenize(frase) for frase in frases]
    #Aplicamos el hidden tager 
    tagged=[]
    for token in tokens:
       tagged.append(tagger.tag(token))
    print(tagged)
    

def procesado_bigram(texto_entrada):
    print("procesado_bigram")
    #Corpus en castellano.
    cess_sents = cess.tagged_sents()
    # Train el Bigram Tagger
    bi_tag = bt(cess_sents[:train])
    print("procesado_bigram")
    
    return 0


def procesado_naive(texto_entrada):
    frases = nltk.sent_tokenize(texto_entrada)
    #sentences = [nltk.word_tokenize(sent) for sent in sentences] 
    #sentences = [nltk.pos_tag(sent) for sent in sentences]
    print("procesado_naive")
    
    return 0


##############################################################################

#Entrenamiento de los tagger
if path.exists('spanish_hmm.plk'):
    hmm_tagger = joblib.load('spanish_hmm.plk')
else:
    #Entrenamos el Hidden tagger
    hmm_tagger = HiddenMarkovModelTagger.train(cess_sents)
    with open('spanish_hmm.plk', 'wb') as pickle_file:
        dill.dump(hmm_tagger, pickle_file) 


if path.exists('unigram.pkl'):
    unigram_tagger = joblib.load('unigram.pkl')
else:
    #Entrenamos el Unigram Tagger
    unigram_tagger = UnigramTagger(cess_sents)
    joblib.dump(unigram_tagger, 'unigram.pkl') 
    
    
if path.exists('bigram.pkl'):
    bigram_tagger = joblib.load('bigram.pkl')  
else:
    #Entrenamos el Bigram Tagger
    bigram_tagger = BigramTagger(cess_sents, backoff=unigram_tagger)
    joblib.dump(bigram_tagger, 'bigram.pkl')     
    

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
        pedido=realizar_pedido(1)

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



