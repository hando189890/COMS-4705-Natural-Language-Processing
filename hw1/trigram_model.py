import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    initial  = []
    if n > 2:
      initial = ['START'] * (n-1) 
    else:
      initial = ['START']

    end = ['STOP']
    
    cur_sequence = initial + sequence + end
    result = []
    
    for i in range(len(cur_sequence) - n + 1):
        result.append(tuple(cur_sequence[i:i+n]))
    
    cur_sequence = ''
    return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.word_count = 0
        self.unigram_count = 0
        self.sentence = 0

        #Your code here
        for sentence in corpus:
            self.unigram_count = self.word_count + len(get_ngrams(sentence, 1)) - 1
            self.word_count = self.word_count + len(sentence) + 1
            self.sentence = self.sentence + 1
            for unigram in get_ngrams(sentence, 1):
               
                if unigram not in self.unigramcounts:
                    self.unigramcounts[unigram] = 1
                else:
                    self.unigramcounts[unigram] = self.unigramcounts[unigram] + 1
    
            for bigram in get_ngrams(sentence, 2):
               
                if bigram not in self.bigramcounts:
                    self.bigramcounts[bigram] = 1
                else:
                    self.bigramcounts[bigram] = self.bigramcounts[bigram] + 1

            for trigram in get_ngrams(sentence, 3):
                
                if trigram not in self.trigramcounts:
                    self.trigramcounts[trigram] = 1
                else:
                    self.trigramcounts[trigram] = self.trigramcounts[trigram] + 1

        # for unigram in self.unigramcounts:
        #     if unigram != ['START'] or unigram != ['STOP']:
        #         self.unigram_count = self.unigram_count + self.unigramcounts.get(unigram)

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        if trigram in self.trigramcounts:
          if trigram[0:2] == ('START','START'):
            return float(self.trigramcounts[trigram]/self.sentence)
          elif trigram[0:2] not in self.bigramcounts.keys():
            return float(self.unigramcounts[trigram[0:1]]/self.unigram_count)
          else:
            return float(self.trigramcounts[trigram]/self.bigramcounts[trigram[0:2]])
        else:
            return 0
    

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram in self.bigramcounts:
            if bigram[:1]==('START',):
                return float(self.bigramcounts[bigram]/self.sentence)
            else:
                return float(self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]])
        else:
            return 0

        # if bigram in self.bigramcounts.keys() and tuple([bigram[0]]) in self.unigramcounts.keys():
        #     return self.bigramcounts[bigram] / self.unigramcounts[tuple([bigram[0]])]
        
           
        return 0

    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        if unigram in self.unigramcounts.keys():
            return self.unigramcounts[unigram] / self.word_count
        return 0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        pass
        # return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3
        lambda2 = 1/3
        lambda3 = 1/3

        temp_tri = trigram
        temp_bi = trigram
        temp_uni = trigram
        return float(lambda1 * self.raw_trigram_probability(temp_tri) + lambda2 * self.raw_bigram_probability(temp_bi[1:]) + lambda3 * self.raw_unigram_probability(temp_uni[2:]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        trigrams = get_ngrams(sentence, 3)
        sentence_logprob = 0
        for trigram in trigrams:
          current_probability = self.smoothed_trigram_probability(trigram)
          sentence_logprob = sentence_logprob + math.log2(current_probability)
        
        return float(sentence_logprob)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0
        M = 0
        total_prob = 0
        for sentence in corpus:
            total_prob = total_prob + self.sentence_logprob(sentence)
            M = M + len(sentence) + 1
       
        l = total_prob/M  
        return  float(pow(2, -l))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        model1 = TrigramModel(training_file1) 
        model2 = TrigramModel(training_file2) 

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1): 
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct = correct+1
            total = total + 1

        for f in os.listdir(testdir2): 
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            if pp2 < pp1:
                 correct = correct+1
        
        return correct / total


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    # print(model)
    

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # cd Desktop
    
    
    # python -i trigram_model.py brown_train.txt brown_test.txt
    # >>>   
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # # Essay scoring experiment: 
    # acc = essay_scoring_experiment("hw1/hw1_data/ets_toefl_data/train_high.txt", 
    #                                "hw1/hw1_data/ets_toefl_data/train_low.txt", 
    #                                "hw1/hw1_data/ets_toefl_data/test_high", 
    #                                "hw1/hw1_data/ets_toefl_data/test_low")
    # print(acc)


    