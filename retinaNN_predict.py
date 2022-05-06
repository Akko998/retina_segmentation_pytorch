###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

# Python
import numpy as np
import configparser

import torch
from matplotlib import pyplot as plt
# scikit learn
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, jaccard_score, f1_score
import sys

from help_functions import *
from extract_patches import *
from pre_processing import my_PreProc
from models.u_net import UNet

# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
# ===========================================
# run the training on invariant or local


device = torch.device("cuda:0" if torch.cuda.is_available() and config.get('testing settings', 'device') == 'gpu'
                      else "cpu")
print(device)
print(torch.__version__)
print(torch.version.cuda)

path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height, full_img_width = test_imgs_orig.shape[2:]

# the border masks provided by the DRIVE
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(DRIVE_test_border_masks)

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

# model name
name_experiment = config.get('experiment name', 'name')
# N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))


# ============ Load the data and divide in patches
patches_imgs_test, test_imgs, test_masks = get_data_testing(
    DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
    DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
    Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
    patch_height=patch_height,
    patch_width=patch_width
)

# ================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')

if best_last == 'best':
    model_file = os.path.join(name_experiment, config.get('experiment name', 'model_best'))
else:
    model_file = os.path.join(name_experiment, config.get('experiment name', 'model_checkpoint'))

checkpoint_params = torch.load(model_file)

# Load the saved model
unet = UNet(1, num_classes=1).to(device, dtype=torch.float)  # the U-net model
unet.load_state_dict(checkpoint_params['model'])
with torch.no_grad():
    patches_pred_test = unet(torch.tensor(patches_imgs_test).to(device, dtype=torch.float)).detach().cpu().numpy()

print("predicted images size :", patches_pred_test.shape)  # (patch, 1, 48 ,48)

# ========== Elaborate and visualize the predicted images ====================
pred_imgs = recompone(patches_pred_test, patch_height, patch_width, full_img_height, full_img_width)  # predictions
pred_imgs[test_border_masks == 0] = 0


## back to original dimensions
print("Orig imgs shape: ", test_imgs.shape)
print("pred imgs shape: ", pred_imgs.shape)
print("Gtruth imgs shape: ", test_masks.shape)
visualize(group_images(test_imgs, N_visual), os.path.join(name_experiment, "all_originals"))  # .show()
visualize(group_images(pred_imgs, N_visual), os.path.join(name_experiment, "all_predictions"))  # .show()
visualize(group_images(test_masks, N_visual), os.path.join(name_experiment, "all_groundTruths"))  # .show()

# visualize results comparing mask and prediction:
assert (test_imgs.shape[0] == pred_imgs.shape[0] and test_imgs.shape[0] == test_masks.shape[0])
N_predicted = test_imgs.shape[0]
group = N_visual
assert (N_predicted % group == 0)
for i in range(int(N_predicted / group)):
    orig_stripe = group_images(test_imgs[i * group:(i * group) + group, :, :, :], group)
    masks_stripe = group_images(test_masks[i * group:(i * group) + group, :, :, :], group)
    pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
    total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
    visualize(total_img,
              os.path.join(name_experiment, name_experiment + "_Original_GroundTruth_Prediction" + str(i)))  # .show()

# ====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
# predictions only inside the FOV
y_scores, y_true = pred_only_FOV(pred_imgs, test_masks, test_border_masks)  # returns data only inside the FOV
print("Calculating results only inside the FOV:")
print("y scores pixels: {} (radius 270: 270*270*3.14==228906), "
      "including background around retina: {} (584*565==329960)".format(y_scores.shape[0], pred_imgs.size))
print("y true pixels: {} (radius 270: 270*270*3.14==228906), "
      "including background around retina: {} (584*565==329960)".format(y_true.shape[0], test_masks.size))

# Area under the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: ", AUC_ROC)
roc_curve = plt.figure()
plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(os.path.join(name_experiment, "ROC.png"))

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision, recall)
print("\nArea under Precision-Recall curve: ", AUC_prec_rec)
prec_rec_curve = plt.figure()
plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(os.path.join(name_experiment, "Precision_recall.png"))

# Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of ", threshold_confusion)
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i] >= threshold_confusion:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)

accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
print("Global Accuracy: ", accuracy)

specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
print("Specificity: ", specificity)

sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
print("Sensitivity: ", sensitivity)

precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
print("Precision: ", precision)

# Jaccard similarity index
jaccard_index = jaccard_score(y_true, y_pred)
print("\nJaccard similarity score: ", jaccard_index)

# F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): ", F1_score)

# Save the results
file_perf = open(os.path.join(name_experiment, 'performances.txt'), 'w')
file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                + "\nJaccard similarity score: " + str(jaccard_index)
                + "\nF1 score (F-measure): " + str(F1_score)
                + "\n\nConfusion matrix:"
                + str(confusion)
                + "\nACCURACY: " + str(accuracy)
                + "\nSENSITIVITY: " + str(sensitivity)
                + "\nSPECIFICITY: " + str(specificity)
                + "\nPRECISION: " + str(precision)
                )
file_perf.close()
