"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        cky_chart = {}
        n = len(tokens)
        #initialization the cky_chart
        for i in range(len(tokens)):
            cky_chart[(i, i+1)] = []
            curent_word = tuple(tokens[i:i+1])
            for rule in self.grammar.rhs_to_rules.get(curent_word, []):
                cky_chart[(i, i+1)].append(rule[0])

        # loop through substrings
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length
                cky_chart[(i, j)] = []
                for k in range(i+1, j):
                    for B in cky_chart.get((i, k), []):
                        for C in cky_chart.get((k, j), []):
                            for rule in self.grammar.rhs_to_rules.get((B, C), []):
                                if rule[0] not in cky_chart[(i, j)]:
                                    cky_chart[(i, j)].append(rule[0])

        # check if sentence is in language
        return self.grammar.startsymbol in cky_chart.get((0, len(tokens)), [])

           
    def parse_with_backpointers(self, tokens):
        
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        table = {}
        probs = {}
        for i in range(len(tokens)):
            for A in self.grammar.lhs_to_rules:
                for s in self.grammar.rhs_to_rules[tuple(tokens[i:i+1])]:
                    if A == s[0]:
                        if (i,i+1) not in probs:
                            probs[(i,i+1)] = {}
                        probs[(i,i+1)][A] = math.log(s[2])
                        if (i,i+1) not in table:
                            table[(i,i+1)] = {}
                        table[(i,i+1)][A] = tokens[i]

        # Main loop
        for length in range(2,len(tokens)+1):
            for i in range(len(tokens)):
                j = i + length
                if j <= len(tokens):
                    for k in range(i + 1, j):
                        if (i,k) in probs and (k,j) in probs:
                            for B in table.get((i,k), []):
                                for C in table.get((k,j), []):
                                    if (B,C) in self.grammar.rhs_to_rules:
                                        for m in self.grammar.rhs_to_rules[(B,C)]:
                                            temp_prob = probs[(i,k)][B] + probs[(k,j)][C] + math.log(m[2])
                                            temp_table_prob = ((m[1][0],i,k), (m[1][1],k,j))
                                            if (i, j) in table:
                                                if m[0] in table[(i, j)]:
                                                    if probs[(i, j)][m[0]] < temp_prob:
                                                        probs[(i, j)][m[0]] = temp_prob
                                                        table[(i, j)][m[0]] = temp_table_prob
                                                else:
                                                    table[(i, j)][m[0]] = temp_table_prob
                                                    probs[(i, j)][m[0]] = temp_prob
                                            
                                            else:
                                                table[(i, j)] = {m[0]: temp_table_prob}
                                                probs[(i, j)] = {m[0]: temp_prob}

        return table, probs



def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if j - i == 1:
        return nt, chart[(i, j)][nt]
        
    rhs = chart[(i,j)][nt]
    if type(rhs) == str:
        return (nt,rhs)
    else:          
        return (nt,get_tree(chart,rhs[0][1],rhs[0][2], rhs[0][0]),get_tree(chart,rhs[1][1],rhs[1][2], rhs[1][0]))
 


 
       
if __name__ == "__main__":

    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
     
       
