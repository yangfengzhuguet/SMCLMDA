import numpy as np
import pandas as pd
import math
import csv



MDA_adj = pd.read_csv('../data/mir2disease+lunwen/-1-0-+1/mirBase_lunwen_101_MDA_.csv', header=None)

MDA_adj=np.array(MDA_adj)
# the number of mRNAs and diseases
nc=np.array(MDA_adj).shape[0]   #535
nd=np.array(MDA_adj).shape[1]   #302



with open("embadding-miRBase+lunwen/-1-0-+1/adj_edgelist.txt", "w") as csvfile:

    for i in range(nc):
        for j in range(nd):
            # if MDA_adj[i][j]==1:
            if MDA_adj[i][j]==1 or MDA_adj[i][j]==-1:
                # csvfile.writerow([i+nd, j])
                csvfile.writelines([str(i+nd), " ", str(j), '\n'])
