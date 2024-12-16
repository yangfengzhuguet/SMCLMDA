import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


# y_true = np.random.randint(0, 2, size=1000)
# y_scores1 = np.random.rand(1000)
# y_scores2 = np.random.rand(1000)
# y_scores3 = np.random.rand(1000)


# fpr1, tpr1, _ = roc_curve(y_true, y_scores1)
# fpr2, tpr2, _ = roc_curve(y_true, y_scores2)
# fpr3, tpr3, _ = roc_curve(y_true, y_scores3)
#
# fpr1= np.load('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_fp1.npy')
# tpr1 = np.load('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_tp1.npy')
# fpr2= np.load('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_fp2.npy')
# tpr2 = np.load('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_tp2.npy')
# fpr3= np.load('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_fp3.npy')
# tpr3 = np.load('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_tp3.npy')
#
#
#
#
# auc1 = auc(fpr1, tpr1)
# auc2 = auc(fpr2, tpr2)
# auc3 = auc(fpr3, tpr3)
#
# # Plotting the ROC curves
# plt.figure(figsize=(8, 6))
# plt.plot(fpr1, tpr1, color='orange', label=f'GAMCNMDF (AUC = {auc1:.4f})')
# plt.plot(fpr2, tpr2, color='green', label=f'GAMCNMDF_G (AUC = {auc2:.4f})')
# plt.plot(fpr3, tpr3, color='blue', label=f'GAMCNMDF_A (AUC = {auc3:.4f})')
#
# # Random classifier line
# plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
#
# # Zoom-in inset
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR (False Positive Rate)')
# plt.ylabel('TPR (True Positive Rate)')
# plt.title('10Fold-CV')
#
# # Add the inset
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# axins = zoomed_inset_axes(plt.gca(), zoom=2, loc='lower right')
# axins.plot(fpr1, tpr1, color='orange')
# axins.plot(fpr2, tpr2, color='green')
# axins.plot(fpr3, tpr3, color='blue')
# axins.set_xlim(0, 0.4)
# axins.set_ylim(0.7, 1.0)
# mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")
#
# # Add the legend
# plt.legend(loc="lower right")
# plt.show()


def visualize_model_with_umap(model, X_test, y_test):
    """
    使用UMAP可视化模型的输入层和最后一层的输出
    """
    # 获取logits层的输出
    logits_model = Model(inputs=model.input, outputs=model.layers[-1].output)  # logits层输出

    # 预测logits输出
    logits_output = logits_model.predict(X_test)

    # 将输入和logits输出转换为UMAP降维后的嵌入表示
    umap_model_input = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    umap_model_fc = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)

    # 降维处理输入特征 (X_test)
    embedding_input = umap_model_input.fit_transform(X_test.reshape(X_test.shape[0], -1))

    # 降维处理logits输出
    embedding_fc = umap_model_fc.fit_transform(logits_output)

    # 绘制输入特征的UMAP
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_input[:, 0], embedding_input[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=10)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('UMAP Visualization of Input Layer')
    plt.savefig('/kaggle/working/Visualization of model inputs(Enhancers stength prediction).png')
    plt.show()

    # 绘制logits输出特征的UMAP
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_fc[:, 0], embedding_fc[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=10)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('UMAP Visualization of FC Output')
    plt.savefig('/kaggle/working/Visualization of model output(Enhancers stength prediction).png')
    plt.show()