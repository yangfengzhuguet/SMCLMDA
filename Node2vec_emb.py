import csv
import numpy as np
import pandas as pd
from ge.node2vec import Node2Vec



import networkx as nx
import os, sys, argparse



def node2vec_emb():

    #   Results obtained from parameter optimization
    pp=2
    qq=0.25

    G = nx.read_edgelist('feature_embedding/embadding-miRBase+lunwen/-1-0-+1/Edgelist_all.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80, p=pp, q=qq, workers=1)    #   , use_rejection_sampling=0
    model.train(embed_size=64, window_size=10, iter=1)
    embeddings = model.get_embeddings()

    ### len(embeddings)=302 + 535
    num_nodes = len(embeddings)
    Gemb_deepwalk = np.zeros([num_nodes, 64])
    # Gemb_deepwalk = np.zeros([len(embeddings), 64])  # num_dimension num_feature

    # for emb in range(len(embeddings)):
    #     for i in range(64):
    #         Gemb_deepwalk[emb][i] = embeddings[str(emb)][i]
    for idx, node in enumerate(embeddings):
        for i in range(64):
            Gemb_deepwalk[idx][i] = embeddings[node][i]
    result = pd.DataFrame(Gemb_deepwalk)
    result.to_csv('emb_node2vec_all.csv', header=False, index=False)




if __name__ == "__main__":
    node2vec_emb()
