import numpy as np
import pandas as pd
import math
import csv
import seaborn as sns


# #################################################################   GIP   ########################################################
#
# dis_gip = pd.read_csv('data_ori/dis_GIPSim.csv', sep=',', header=None)
# circ_gip = pd.read_csv('data_ori/circ_GIPSim.csv', sep=',', header=None)
# # dis_gip = pd.read_csv('../Dataset2/disease_GIPsimilarity.csv', sep=',', header=None)
# # circ_gip = pd.read_csv('../Dataset2/circRNA_GIPsimilarity.csv', sep=',', header=None)
#
#
# # Modify the similarity feature matrix to a 0/1 binary association matrix based on the set threshold
# dis_gip, n1 = get_new_matrix_MD(dis_gip, 0.5)
# circ_gip, n2 = get_new_matrix_MD(circ_gip, 0.5)
#
# print("The number of dis interaction is %s and circRNA inseraction is %s in Heterogeneous graph by GIP Sim" %(n1,n2))
#
#
# dis_gip=np.array(dis_gip)
# circ_gip=np.array(circ_gip)
# nd=np.array(dis_gip).shape[0]   #88
# nc=np.array(circ_gip).shape[0]   #585
#
#
# with open("Dataset1/adj_eagelist_GIP.txt", "w") as csvfile:  #   ../Dataset2/adj_eagelist_GIP.txt
#     for i in range(nd):
#         for j in range(i):
#             if dis_gip[i][j]==1:
#                 csvfile.writelines([str(i)," ",str(j),'\n'])
#
#     for i in range(nc):
#         for j in range(i):
#             if circ_gip[i][j]==1:
#                 csvfile.writelines([str(i+nd), " ", str(j+nd), '\n'])
#                 # writer.writerow([i+88, j+88])
#
#
# #################################################################   Cosine    ########################################################
#
# dis_gip = pd.read_csv('data_ori/dis_CosSim.csv', sep=',', header=None)
# circ_gip = pd.read_csv('data_ori/circ_CosSim.csv', sep=',', header=None)
#
# # Modify the similarity feature matrix to a 0/1 binary association matrix based on the set threshold
# dis_gip, n1 = get_new_matrix_MD(dis_gip, 0.5)
# circ_gip, n2 = get_new_matrix_MD(circ_gip, 0.5)
#
# print("The number of dis interaction is %s and circRNA inseraction is %s in Heterogeneous graph by Cosine Sim" %(n1,n2))
#
#
# dis_gip=np.array(dis_gip)
# circ_gip=np.array(circ_gip)
# nd=np.array(dis_gip).shape[0]   #88
# nc=np.array(circ_gip).shape[0]   #585
#
#
# with open("Dataset1/adj_eagelist_cosine.txt", "w") as csvfile:
#     # writer = csv.writer(csvfile)
#
#     for i in range(nd):
#         for j in range(i):
#             if dis_gip[i][j]==1:
#                 csvfile.writelines([str(i), " ", str(j), '\n'])
#
#     for i in range(nc):
#         for j in range(i):
#             if circ_gip[i][j]==1:
#                 csvfile.writelines([str(i + nd), " ", str(j + nd), '\n'])


def get_new_matrix_MD(D_D, th_d):
    N_d = D_D.shape[0]
    d = np.zeros([N_d, N_d])
    tep_d = 0
    for i in range(N_d):
        for j in range(i):
            # if D_D[i][j] > th_d or D_D[j][i] > th_d:
            if D_D[i][j] > 0.5 or D_D[j][i] > 0.5:
                d[i][j] = 1
                d[j][i] = 1
                tep_d = tep_d + 1
            elif D_D[i][j] < 0 or D_D[j][i] < 0:
                d[i][j] = 1
                d[j][i] = 1
                tep_d = tep_d + 1
            else:
                d[i][j] = 0
    return d, tep_d
# ################################################# Ensemble Similarity  Sim_all   ########################################################
dis_all = pd.read_csv('embadding-miRBase+lunwen/-1-0-+1/disSim_all.csv', sep=',', header=None)
mi_all = pd.read_csv('embadding-miRBase+lunwen/-1-0-+1/miSim_all.csv', sep=',', header=None)
# dis_th = threshold(dis_all) # When converting to a 0/1 matrix, first select the threshold
# mi_th = threshold(mi_all) # When converting to a 0/1 matrix, first select the threshold
# print(dis_th)
# print(mi_th)

# Modify the similarity feature matrix to a 0/1 binary association matrix based on the set threshold
dis_all, n1 = get_new_matrix_MD(dis_all, 0.5)
mi_all, n2 = get_new_matrix_MD(mi_all, 0.5)
print("The number of dis interaction is %s and miRNA inseraction is %s in Heterogeneous graph by intergate Sim" %(n1,n2))


dis_all=np.array(dis_all)
mi_all=np.array(mi_all)
nd=np.array(dis_all).shape[0]   #302
nm=np.array(mi_all).shape[0]   #535


with open("embadding-miRBase+lunwen/-1-0-+1/adj_eagelist_all.txt", "w") as csvfile:
    for i in range(nd):
        for j in range(i):
            if dis_all[i][j]==1:
                csvfile.writelines([str(i)," ",str(j),'\n'])

    for i in range(nm):
        for j in range(i):
            if mi_all[i][j]==1:
                csvfile.writelines([str(i+nd), " ", str(j+nd), '\n'])
