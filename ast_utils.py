from pycparser import c_parser, c_ast
import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
import pickle
#from tree import ASTNode, SingleNode
import numpy as np
from clang.cindex import CursorKind

class Node:
    displayname=None
    kind = None
    def __init__(self, name):
         self.displayname = name
         self.kind=name

def get_sequences(node, sequence):
    kind=str(node.kind)[11:]      # remove CursorKind. from beginning
    sequence.append(kind)
    if not node.displayname == '':
        sequence.append(node.displayname)
    for child in node.get_children():
        get_sequences(child, sequence)
    if node.kind == CursorKind.COMPOUND_STMT:
        sequence.append('End')


def get_blocks(node, block_seq):
    children = node.get_children()
    num_children=0
    for child in children:
        num_chuildren = num_children +1
    name = str(node.kind)[11:]
    print(name)   # will show all the ast node type
    if name in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
        block_seq.append(node)
        if name is not 'FOR_STMT':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, num_children):
            child = children[i][1]
            if child.kind not in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT', 'COMPOUND_STMT']:
                block_seq.append(child)
            get_blocks(child, block_seq)
    elif name is 'COMPOUND_STMT':
        block_seq.append(Node(name))
        for child in node.get_children():
            if child.kind not in ['IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
                block_seq.append(child)
            get_blocks(child, block_seq)
        block_seq.append(Node('End'))  # block_seq contains node End should be a node
    else:
        for child in node.get_children():
            get_blocks(child, block_seq)
























