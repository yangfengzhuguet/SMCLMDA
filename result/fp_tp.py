import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc

# mean_fpr = np.linspace(0, 1, 100)
#
#
# tpr_list = []
#
#
# for i in range(10):
#     fpr = np.load(f'HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_fp{i + 1}.npy')
#     tpr = np.load(f'HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_tp{i + 1}.npy')
#
#
#     tpr_interp = np.interp(mean_fpr, fpr, tpr)
#     tpr_list.append(tpr_interp)
#
#
# mean_tpr = np.mean(tpr_list, axis=0)
#
#
# np.save('HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_final_fpr.npy', mean_fpr)
# np.save('HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_final_tpr.npy', mean_tpr)
#
#
# print(f'Mean FPR shape: {mean_fpr.shape}')
# print(f'Mean TPR shape: {mean_tpr.shape}')


fpr = np.load('HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_final_fpr.npy')
tpr = np.load('HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_final_tpr.npy')
auroc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()