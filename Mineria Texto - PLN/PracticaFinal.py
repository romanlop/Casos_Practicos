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


#Defición de funciones.
def realizar_pedido():
    pedido=1000
    while 1:
        print("Buenos días. ¿Cuál es su pedido?. Introduzca exit para salir.")
        pedido=input()
        if pedido != "exit":
            if int(option)==1:
                print("Opción 1. Regex Parser")
                procesado_regex(pedido)
        
            elif int(option)==2:
                print("Opción 2. Unigram tagger")
                procesado_unigram(pedido)
        
            elif int(option)==3:
                print("Opción 3. Bigram Tagger")
    
                
            elif int(option)==4:
                print("Opción 4. NaiveBayes Classifier")
        else:
            break

    return pedido


def procesado_regex(texto_entrada):
    frases = nltk.sent_tokenize(texto_entrada)
    tokens = [nltk.word_tokenize(frase) for frase in frases]
    tagged = [hmm_tagger.tag(token) for token in tokens]
    print ("TAGGER:",tagged)
    #definimos la gramática.
    grammar = "NP: {(<di0ms0>|<dn0cp0>|<pi0ms000>|<di0fs0>)+(<ncms000>|<ncmp000>|<ncfs000>)+}"
    cp = nltk.RegexpParser(grammar)
    chunked = []
    for s in tagged:
        result=cp.parse(s)
    print(result)
    result.draw()
    #Guardar esto en formato IOB para entrenar los otros corpus...
    return 0
    



def procesado_unigram(texto_entrada):
    print("procesado_unigram")
    #texto_entrada = "Hola. Buenos días. Quería una pizza con queso. Mi domilicio está en Santiago."
    frases = nltk.sent_tokenize(texto_entrada)
    tokens = [nltk.word_tokenize(frase) for frase in frases]
    tagged = [unigram_tagger.tag(token) for token in tokens]
    print ("TAGGER UNIGRAMA CAT:",tagged)
    return 0


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
    

"""
Descomentar para entrar los modelos Unigram y Bigram con cess_sents.
"""
#Corpus en castellano.
cess_sents = cess.tagged_sents()

if path.exists('spanish_hmm.plk'):
    hmm_tagger = joblib.load('spanish_hmm.plk')
else:
    #Entrenamos el Unigram Tagger
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
    #Entrenamos el Unigram Tagger
    bigram_tagger = BigramTagger(cess_sents, backoff=unigram_tagger)
    joblib.dump(bigram_tagger, 'bigram.pkl')     
    


#Menú principal
print("Selección del modo de funcionamiento:")
print("1. RegexParser")
print("2. Unigram tagger")
print("3. Bigram Tagger")
print("4. NaiveBayes Classifier")
option = input()

if int(option)==1:
    print("Ha seleccionado la opción 1. RegexParser")
    pedido=realizar_pedido()
    exit(0)
    
elif int(option)==2:
    print("Opción 2. Unigram tagger")
    pedido=realizar_pedido()
    exit(0)
    
elif int(option)==3:
    print("Opción 3. Bigram Tagger")
    pedido=realizar_pedido()
    exit(0)
    
elif int(option)==4:
    print("Opción 4. NaiveBayes Classifier")
    pedido=realizar_pedido()
    exit(0)
    
else:
    exit(0)
    





