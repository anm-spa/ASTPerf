import pandas as pd
import torch
import time
import numpy as np
import warnings
import tracemalloc
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')



def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['ast_x'])
        x2.append(item['ast_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


if __name__ == '__main__':
    tracemalloc.start(25)
    root = 'Data/'
    train_data = pd.read_pickle(root+'train/blocks.pkl').sample(frac=1)
    test_data = pd.read_pickle(root+'test/blocks.pkl').sample(frac=1)
    val_data = pd.read_pickle(root + 'dev/blocks.pkl').sample(frac=1)
    categories = 1
    print("Training...")

    word2vec = Word2Vec.load(root+"/train/ast_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 32
    USE_GPU = False

    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    for t in range(1, categories+1):
        
        train_data_t, test_data_t = train_data, test_data
        # training procedure
        for epoch in range(EPOCHS):
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
             #   from numpy import array
             #   a = array(train1_inputs)
             #   print(a.shape)
             #   lens = [len(item) for item in train1_inputs]
             #   print(lens)
               # train1_inputsZ = train1_inputs[0]
               # train2_inputsZ = train2_inputs[0]
               # a = array(train1_inputsZ)
               # print(a.shape)
               # lens = [len(item) for item in train1_inputs]
               # print(lens)
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('traceback')
                stat = top_stats[0]
                print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
                for line in stat.traceback.format():
                    print(line)
        print("Testing-%d..."%t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test1_inputs, test2_inputs)

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
