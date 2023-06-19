#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

import math


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    # return [] Function is writen in google colab, please see lexsub_main.ipynb for all procedurals
    possible_synonyms = []
    for synset in wn.synsets(lemma, pos):
        for synonym in synset.lemmas():
            if synonym.name() != lemma: 
              if "_" not in synonym.name():
                  possible_synonyms.append(synonym.name())
              elif "_" in synonym.name():   
                  possible_synonyms.append(synonym.name().replace('_',' '))
    possible_synonyms = list(set(possible_synonyms)) #remove duplicate
  
    return possible_synonyms

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # return None # replace for part 2
    #Function is writen in google colab, please see lexsub_main.ipynb for all procedurals
    word_dict = {} 
    for lemma in wn.lemmas(context.lemma, pos=context.pos):
        for related_lemma in lemma.synset().lemmas():
            if related_lemma.name().lower() != context.lemma:
                word_dict[related_lemma.name().lower().replace('_',' ')] = word_dict.get(related_lemma.name().lower().replace('_',' '), 0) + related_lemma.count()
     
    return max(word_dict, key=word_dict.get) if len(word_dict) !=0 else None


def wn_simple_lesk_predictor(context : Context) -> str:
    # return None #replace for part 3   
    #Function is writen in google colab, please see lexsub_main.ipynb for all procedurals  
    stopwords_set = set(stopwords.words('english'))
    leftright = [x.lower() for x in context.left_context + context.right_context]
    leftright_clear = []
    i = 0
    while i < len(leftright):
        if leftright[i] not in stopwords_set:
            leftright_clear.append(leftright[i])
        i += 1
    
    max_score = 0
    result = None
    synlist = set()
    

    for l1 in wn.lemmas(context.lemma, pos=context.pos):
        overlap = set()
        score = 0
        syn = l1.synset()
        
        # Get all example and definition sentences
        df = syn.examples()[:]
        df.append(syn.definition())
        for hyper in syn.hypernyms():
            df.append(hyper.definition())
            df += hyper.examples()
        
        
        for sentence in df:
            synlist |= set(tokenize(sentence.lower()))
        
        for word in synlist:
            if word in leftright_clear and word not in overlap:
                overlap.add(word)
 
        for lem in syn.lemmas():
            if lem.name().lower() != context.lemma:
                score = 1000*len(overlap) + 100*l1.count() + lem.count()
                if score >= max_score:
                    max_score = score
                    result = lem.name()
                    result = result.replace('_', ' ')
    
    return result  


def wn_refined_predictor(context: Context) -> str:
    #part 6 Change made compared with wn_simple_lesk_predictor in part 3
    #Changed the way the score is calculated. Instead of adding up the counts, I used an exponential function to give more weight to the overlap between the context and the synset.
    #Used Python's built-in set data type to eliminate duplicate words in the synset and the overlap, rather than using lists and manually checking for duplicates.
    #Removed unnecessary variable assignments and redundant code.
    
    #Function is writen in google colab, please see lexsub_main.ipynb for all procedurals
    stopwords_set = set(stopwords.words('english'))
    leftright = [x.lower() for x in context.left_context + context.right_context]
    leftright_clear = [x for x in leftright if x not in stopwords_set]
    
    max_score = 0
    result = None
    synlist = set()
    
    for l1 in wn.lemmas(context.lemma, pos=context.pos):
        overlap = set()
        score = 0
        syn = l1.synset()
        
        # Get all example and definition sentences
        df = syn.examples()[:]
        df.append(syn.definition())
        for hyper in syn.hypernyms():
            df.append(hyper.definition())
            df += hyper.examples()
        
        
        for sentence in df:
            synlist |= set(tokenize(sentence.lower()))
        
        for word in synlist:
            if word in leftright_clear and word not in overlap:
                overlap.add(word)
 
        for lem in syn.lemmas():
            if lem.name().lower() != context.lemma:
                score = len(overlap) + math.exp(l1.count()) + math.exp(lem.count())
                if score > max_score:
                    max_score = score
                    result = lem.name().replace('_', ' ')
    
    return result 
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # return None # replace for part 4

        #Function is writen in google colab, please see lexsub_main.ipynb for all procedurals
        candidates = get_candidates(context.lemma, context.pos)
        max_score = -1
        result = None
        for candidate in candidates:
            try:
                score = self.model.similarity(context.lemma, candidate)
                if score > max_score:
                    max_score = score
                    result = candidate
            except KeyError:
                continue
        return result



class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # return None # replace for part 5

        #Function is writen in google colab, please see lexsub_main.ipynb for all procedurals
        candidates = get_candidates(context.lemma, context.pos)
        inputlist = context.left_context[:] + ['[MASK]'] + context.right_context[:]
        input_toks = self.tokenizer.encode(inputlist)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=None)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][len(self.tokenizer.encode(context.left_context))-1])[::-1]
        wordcand = self.tokenizer.convert_ids_to_tokens(best_words)
        for word in wordcand:
            if word in candidates:
                return word
        return wordcand[0] 

    

if __name__=="__main__":

    #Functions is writen in google colab, please see lexsub_main.ipynb or lexsub_main.pdf for all procedurals and results

    # At submission time, this program should run your best predictor (part 6).
    #The best predictor is part 5 which is the distilbert-base-uncased bert language model from hugging face 

    bert_predictor = BertPredictor()

    # read lexsub_trial.xml file and apply smurf_predictor function to each context
    for context in read_lexsub_xml(sys.argv[1]):
        prediction = bert_predictor.predict(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))


    #######################################################

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # for context in read_lexsub_xml(sys.argv[1]):
    #     #print(context)  # useful for debugging
    #     prediction = smurf_predictor(context) 
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
