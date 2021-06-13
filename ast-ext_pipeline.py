#! /usr/bin/python

import clang.cindex
import sys
import pandas as pd
import os
from utils import insert

clang.cindex.Config.set_library_path("/Users/mar01/DevTools/llvm-xcodeNew/llvm-project/build/Debug/lib/")
index = clang.cindex.Index.create()
#path="Data/Commit-Diffs-Preprocess/"
path=os.getcwd() + '/' + "Data/Commit-Diffs-Originals/"
dpath=os.getcwd() + '/' + "Data/"

#gitDir="/Users/mar01/DevTools/git"
#repo = git.Repo("/Users/mar01/DevTools/git")
#sha = repo.head.object
Fin='Data/ext-labels-updated10.csv'
labelled_data=pd.read_csv(Fin, low_memory = True)
labelled_data = labelled_data.set_index(['commA','commB'])

class ASTanalysis:
    def __init__(self,  ratio, root):
        self.ratio = ratio
        self.root = root
        self.asts = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.train=None
        self.dev=None
        self.test=None
        self.size = None
        self.ast=None
        self.labels=None
        self.w2vec=None
        self.blocks=None
       # self.w2vecOptimized=None
        
    def tobeparsedAB_tus(self):
        columns=['idA','idB','files_A','files_B','labels']
        df0 = pd.DataFrame(columns=columns)
        k=0;
        if os.path.isdir(path):
            for dEntry in os.listdir(path):
                k=k+1
                if k>25:
                    break
                if os.path.isdir(path+dEntry):
                    d1,d2=os.listdir(path+dEntry)[:2]
                    _q1, idA = os.path.split(d1)
                    _q2, idB = os.path.split(d2)
                    Fs1=''
                    Fs2=''
                    if os.path.isdir(path+dEntry+'/'+d1):
                            if os.path.isfile(path+dEntry+'/'+d1+'/main.c'):
                                Fs1=path+dEntry+'/'+d1+'/main.c'
                    if os.path.isdir(path+dEntry+'/'+d2):
                            if os.path.isfile(path+dEntry+'/'+d2+'/main.c'):
                                Fs2=path+dEntry+'/'+d2+'/main.c'
                    _p, id = os.path.split(dEntry)
                    try:
                        l=labelled_data.at[(idA,idB),'labels']
                    except:
                        try:
                            l=labelled_data.at[(idB,idA),'labels']
                        except:
                            print("Value not found: ",idA,idB)
                            l=-1
                    idx=id + ':' + idA
                    idy=id + ':' + idB
                    if Fs1!= '' and Fs2!= '':
                        insert(df0,[idx,idy,Fs1,Fs2,l])
        return df0
    
    def tobeparsed_tus(self):
        columns=['id','code']
        df0 = pd.DataFrame(columns=columns)
        columns=['idA','idB','label']
        df1 = pd.DataFrame(columns=columns)
        k=0;
        if os.path.isdir(path):
            for dEntry in os.listdir(path):
                k=k+1
                if k>25:
                    break
                if os.path.isdir(path+dEntry):
                    dirs=[]
                    for d in os.listdir(path+dEntry):
                 #       print(d)
                        if os.path.isdir(path+dEntry+'/'+d):
                            dirs.append(d)
                 #   print(dirs)
                    d1,d2=dirs[:2]
                  #  d1,d2=os.listdir(path+dEntry)[:2]
                    _q1, idA = os.path.split(d1)
                    _q2, idB = os.path.split(d2)
                    Fs1=''
                    Fs2=''
                    if os.path.isdir(path+dEntry+'/'+d1):
                        if os.path.isfile(path+dEntry+'/'+d1+'/main.c'):
                            Fs1=path+dEntry+'/'+d1+'/main.c'
                    if os.path.isdir(path+dEntry+'/'+d2):
                        if os.path.isfile(path+dEntry+'/'+d2+'/main.c'):
                            Fs2=path+dEntry+'/'+d2+'/main.c'
                    _p, id = os.path.split(dEntry)
                    try:
                        l=labelled_data.at[(idA,idB),'labels']
                    except:
                        try:
                            l=labelled_data.at[(idB,idA),'labels']
                        except:
                            print("Value not found: ",idA,idB,dEntry)
                            l=-1
                    idx=id + ':' + idA
                    idy=id + ':' + idB
                    if Fs1!= '' and Fs2!= '':
                        insert(df0,[idx,Fs1])
                        insert(df0,[idy,Fs2])
                        insert(df1,[idx,idy,l])
            return (df0,df1)
    # parse source code
    def parse_source(self, output_file, option):
        path = self.root+output_file
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            
            (df0,df1)=self.tobeparsed_tus()
            #columns=['id', 'sha','asts']
            columns=['id','ast']
            self.asts = pd.DataFrame(columns=columns)
            for i, row in df0.iterrows():
                ast=index.parse(row["code"])
                insert(self.asts,[row["id"],ast.cursor])
            self.labels=df1
        
        # split data for training, developing and testing
    def split_data(self):
        data = self.labels
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = self.root+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        #train.to_pickle(self.train_file_path)
        self.train=train

        dev_path = self.root+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        #dev.to_pickle(self.dev_file_path)
        self.dev=dev

        test_path = self.root+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        #test.to_pickle(self.test_file_path)
        self.test=test
            
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
            
        pairs=self.train
        train_ids = pairs['idA'].append(pairs['idB']).unique()
        #print(train_ids)

        trees = self.asts.set_index('id',drop=False).loc[train_ids]
        from ast_utils import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
          #  print(sequence)
            return sequence
                       
        corpus = trees['ast'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        #print(str_corpus)

        #FH=open("corpus.txt","w")
        #for w in str_corpus:
        #    FH.write(w)
        #FH.close()
        
        trees['ast'] = pd.Series(str_corpus)
        trees.to_csv(dpath+'/train/programs_ns.tsv')
        #print(trees['ast'])
        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
       # print("Beginning print")
       # print(corpus)
       # print("Corpus already printed")
        w2v.save(self.root+'train/ast_w2v_' + str(size))
        self.w2vec=w2v
                
        
    # generate block sequences with index representations
    def generate_block_seqs(self):
        from ast_utils import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        #word2vec = Word2Vec.load(self.root+'train/node_w2v_' + str(self.size)).wv
        vocab = self.w2vec.wv.vocab
        max_token = self.w2vec.wv.vectors.shape[0]
       
        def tree_to_index(node):
            token = node.displayname
            if token!="":
                print(token)
                result = [vocab[token].index if token in vocab else max_token]
            else:
                name = str(node.kind)[11:]
                print(name)
                result = [vocab[name].index if name in vocab else max_token]
            
            children = node.get_children()
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            #print("Printing blocks...")
            #print(blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees=pd.DataFrame(self.asts, copy=True)
        trees['ast']=trees['ast'].apply(trans2seq)
        self.blocks=trees
        #print(trees)
        
    def merge(self,data_path,part):
        if part == 'train':
            pairs=self.train
          #  print(self.train)
        elif part == 'dev':
            pairs=self.dev
        else:
            pairs=self.test
        pairs=self.labels
        pairs['idA'] = pairs['idA']
        pairs['idB'] = pairs['idB']
        df = pd.merge(pairs, self.blocks, how='left', left_on='idA', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='idB', right_on='id')
        df.drop(['idA', 'idB'], axis=1,inplace=True)
        df.dropna(inplace=True)
        
        #from numpy import array
        #a = array(df)
        #print(a.shape)
        df.to_pickle(self.root+'/'+part+'/blocks.pkl')
        
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        #print("Parsing Done...")
        #print(self.asts)
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')
        
analyzer=ASTanalysis('3:1:1','Data/')
analyzer.run()

#walk(ast.cursor)
