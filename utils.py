###########  @Abu Naser Masud ##########
# Remove irrelevant infromation from the CSV files and generate new CSV data
import sys
import os
import pathlib
import fnmatch
import git
import subprocess
import pandas as pd
import pickle
 
#path="Data/Commit-Diffs/"
#Fin='Data/ext-labels10.csv'
#df=pd.read_csv(Fin, low_memory = True)
#currDir=os.getcwd()
#columns=['id', 'sha','code','labels']
#df = pd.DataFrame(columns=columns)
#FileName=currDir+'/Data/commit-diff.csv'

def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row

if __name__ == '__main__':
    full_path=currDir + '/'+path
    if os.path.isdir(full_path):
        #Dirs=os.scandir(full_path)
        for dEntry in os.listdir(full_path):
            #ds=os.scandir(full_path+dEntry)
            #print(dEntry)
            if os.path.isdir(full_path+dEntry):
                print(full_path+dEntry)
                for d in os.listdir(full_path+dEntry):
                    print(full_path+dEntry+'/'+d)
                    if os.path.isdir(full_path+dEntry+'/'+d):
                        Fs=[]
                        #Files=os.scandir(full_path+ dEntry + '/'+ d)
                        for file in os.listdir(full_path+ dEntry + '/'+ d):
                            print(file)
                            if os.path.isfile(full_path+dEntry+'/'+d+'/'+file):
                                Fs.append(full_path+dEntry+'/'+d+'/'+file)
                    insert(df,[dEntry,d,Fs,0])
    df.to_csv(FileName)
    with open('Data/commit-diff.pickle', 'wb') as f:
        pickle.dump(df, f)
   
