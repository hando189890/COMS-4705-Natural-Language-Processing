from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            # pass
            # TODO: Write the body of this loop for part 4 
            features = self.extractor.get_input_representation(words, pos, state).reshape(1,6)
            possible_actions = self.model.predict(features)[0]
            sorted_actions =  list(np.argsort(possible_actions)[::-1])
            valid_transition = False
            for index in sorted_actions:
                if self.output_labels[index][0] == "shift":
                    if len(state.buffer) > 1 or len(state.stack)==0 :
                        state.shift()
                        valid_transition = True
                elif self.output_labels[index][0] == "left_arc":
                    if len(state.stack) !=0 and state.stack[-1] != 0:
                        state.left_arc(self.output_labels[index][1])
                        valid_transition = True
                elif self.output_labels[index][0] == "right_arc":
                    if len(state.stack) !=0 :
                        state.right_arc(self.output_labels[index][1])
                        valid_transition = True
                if valid_transition:
                    break

            if not valid_transition:
                raise ValueError("No valid transition available.")    


        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 

        #Micro Avg. Labeled Attachment Score: 0.6839216271568701
        #Micro Avg. Unlabeled Attachment Score: 0.7436630929578866

        #Macro Avg. Labeled Attachment Score: 0.6920165164284199
        #Macro Avg. Unlabeled Attachment Score: 0.7515093444830464
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
