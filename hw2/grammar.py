"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
import math
from math import fsum


class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
 
        for lhs in self.lhs_to_rules:
            if lhs == 'PUN':
                # check base case punctuation, 
                # although there is a post in ed, i think it is better to check
                rhs = self.lhs_to_rules[lhs]
                if not (len(rhs[0][1]) != 1 or rhs[0][1][0] == '.' or rhs[0][2] == 1.0):
                    return False
            
            sum_probs = 0
            for rhs in self.lhs_to_rules[lhs]:
                if len(rhs[1]) == 1:
                    # check base case single terminals
                    cur_rhs = rhs[1][0]
                    if not cur_rhs.lower() == rhs[1][0]:
                        return False
                    else:
                        sum_probs += rhs[2]
                elif len(rhs[1]) == 2:
                    # check if non-terminals, rhs should both be uppercase
                    if (rhs[1][0].isupper() == False) or (rhs[1][1].isupper() == False):
                        return False
                    else:
                        sum_probs += rhs[2]
                else:
                    # rule length should be 1 or 2, check if invalid rule length
                    return False

            if lhs.isupper() == False:
                # check uppercase non-terminals
                return False
            if not math.isclose(sum_probs, 1):
                # check sum of probabilities for same lhs, should close to 1.0
                return False
            
        return True



if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        # print(grammar.verify_grammar())



    # with open('atis3.pcfg','r') as grammar_file: 
    #     grammar = Pcfg(grammar_file)
    #     print(grammar.verify_grammar())

     

    

   
