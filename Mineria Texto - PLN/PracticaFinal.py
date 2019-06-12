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

###############################################################################

#Corpus en castellano.
cess_sents = cess.tagged_sents()


#Frases de ejemplo
corpus_ejemplo=['Buenos días. Quiero una pizza y quiero dos pollos.',
                'Buenos días. Quiero tres pollos y  dos pasteles.',
                'Hola. Me gustaria comer cuatro helados y una paella.',
                'Me gustaria comer cuatro filetes y dos ensaladas. También quiero un helado.']

#Tiene que ser la ruta completa
corpus_file='/Users/Ruman/Desktop/DOC Master/Repositorio GITHUB/Casos_Practicos/Mineria Texto - PLN/corpus/corpus_regex.txt'

#Corpus que vamos a crear con REXPARSER.
corpus_comida=""

###############################################################################


#Cambiarlo para meter frases de ejemplo!!!!


#Defición de funciones.
def realizar_pedido():
    for pedido in corpus_ejemplo:
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


#Crea el fichero de corpus.
def procesado_regex(texto_entrada):
    #Separamos en frases.
    frases = nltk.sent_tokenize(texto_entrada)
    #Tokenizamos.
    tokens = [nltk.word_tokenize(frase) for frase in frases]
    #Aplicamos el hidden tager 
    tagged = [hmm_tagger.tag(token) for token in tokens]
    #print ("TAGGER:",tagged)
    #definimos la gramática.
    #grammar = "NP: {(<di0ms0>|<dn0cp0>|<pi0ms000>|<di0fs0>)+(<ncms000>|<ncmp000>|<ncfs000>)+}"
    #MEJORARLO CON NÚMEROS Y DEMÁS...
    cp = nltk.RegexpParser('''
                           COMIDA: {(<ncms000>|<ncmp000>|<ncfs000>)+}
                           CANTIDAD: {(<di0ms0>|<dn0cp0>|<pi0ms000>|<di0fs0>)+}
                           ''')
    chunked = []
    #Aplicamos Regexparses sobre nuestros tokens tageados.
    for s in tagged:
        result=cp.parse(s)
    #print(result)
    #result.draw()
    iob_tags = tree2conlltags(result)
    #print (iob_tags)
    #Guardar esto en formato IOB para entrenar los otros corpus...
    print (iob_tags)
    with open(corpus_file, 'a+') as f: 
        for item in iob_tags:
            for l in item:
                f.write(str(l))
                f.write(" ")
            f.write("\n")
        f.write(". . O")
        f.write("\n")
        f.write("\n")
    f.close()
    return 0
    


def parser_IOB(fichero):
    corpus_comida=conll2000.chunked_sents(fichero, chunk_types=['NP'])
    return corpus_comida

"""TODO. ENTRENAR EL UNIGRAM o BIGRAM CON EL CORPUS ANTERIOR QUE ME CURRE.
PROBAR CON COSAS QUE NO SEAN COMIDA. PILLAR EL CÓDIGO DE EJEMPLO.
"""

def procesado_unigram(texto_entrada):
    corpus = parser_IOB(corpus_file)
    print(corpus[1])
    """print("procesado_unigram")
    #Leer el fichero de corpus
    f=open(corpus_file, "r")
    contents = f.read()
    corpus_comida = nltk.chunk.conllstr2tree(contents, chunk_types=['NP'])
    print(corpus_comida)
    #texto_entrada = "Hola. Buenos días. Quería una pizza con queso. Mi domilicio está en Santiago."
    #frases = nltk.sent_tokenize(texto_entrada)
    #tokens = [nltk.word_tokenize(frase) for frase in frases]
    #tagged = [unigram_tagger.tag(token) for token in tokens]
    #print ("TAGGER UNIGRAMA CAT:",tagged)"""
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
print("Selección del modo de funcionamiento:")
print("1. RegexParser")
print("2. Unigram tagger")
print("3. Bigram Tagger")
print("4. NaiveBayes Classifier")
print("Otro para salir.")
option = input()

if int(option)==1:
    print("Ha seleccionado la opción 1. RegexParser")
    pedido=realizar_pedido()
    #exit(0)
    
elif int(option)==2:
    print("Opción 2. Unigram tagger")
    pedido=realizar_pedido()
    
elif int(option)==3:
    print("Opción 3. Bigram Tagger")
    pedido=realizar_pedido()
    
elif int(option)==4:
    print("Opción 4. NaiveBayes Classifier")
    pedido=realizar_pedido()
    
else:
    exit(0)
    


#Recuperar Arbol -> tree = conlltags2tree(iob_tags)
#print tree


