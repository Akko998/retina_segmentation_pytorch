import math

import numpy as np
import random
import configparser

import torch

from help_functions import load_hdf5
from pre_processing import my_PreProc

# To select the same images
random.seed(10)


# Load the original data and return the extracted patches for training/testing
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV,
                      vali_prop=0.1):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth)  # masks always the same
    # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    # preprocessing
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.

    # cliping
    train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntrain images/masks shape:", train_imgs.shape)  # (20, 1, 565, 565)
    print("train images range (min-max): {} - {}".format(np.min(train_imgs), np.max(train_imgs)))
    print("train masks are within 0-1\n")

    # extract the patches from the full images
    patches_imgs, patches_masks = extract_random(train_imgs, train_masks, patch_height, patch_width,
                                                 N_subimgs, inside_FOV)
    data_consistency_check(patches_imgs, patches_masks)

    # split training & vali
    spl_ind = int((1 - vali_prop) * N_subimgs)
    patches_imgs_train, patches_imgs_vali = patches_imgs[:spl_ind, :, :, :], patches_imgs[spl_ind:, :, :, :]
    patches_masks_train, patches_masks_vali = patches_masks[:spl_ind, :, :, :], patches_masks[spl_ind:, :, :, :]

    print("train PATCHES images shape:", patches_imgs_train.shape)  # (171000, 1, 48, 48)
    print("train PATCHES masks shape:", patches_masks_train.shape)  # (171000, 1, 48, 48)
    print("train PATCHES images range (min-max): {} - {}".format(np.min(patches_imgs_train), np.max(patches_imgs_train)))

    print("validation PATCHES images shape:", patches_imgs_vali.shape)  # (19000, 1, 48, 48)
    print("validation PATCHES masks shape:", patches_masks_vali.shape)  # (19000, 1, 48, 48)
    print("validation PATCHES images range (min-max): {} - {}".format(np.min(patches_imgs_vali), np.max(patches_imgs_vali)))

    return patches_imgs_train, patches_masks_train, patches_imgs_vali, patches_masks_vali


# Load the original data and return the extracted patches for training/testing
def get_data_testing(DRIVE_test_imgs_original,
                     DRIVE_test_groudTruth,
                     Imgs_to_test,
                     patch_height,
                     patch_width):
    ### test
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.

    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    test_masks = test_masks[0:Imgs_to_test, :, :, :]

    data_consistency_check(test_imgs, test_masks)

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)

    print("\ntest images/masks shape:", test_imgs.shape)
    print("test images range (min-max): ", np.min(test_imgs), ' - ', np.max(test_imgs))
    print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs, patch_height, patch_width)

    print("\ntest PATCHES images shape:", patches_imgs_test.shape)
    print("test PATCHES images range (min-max): {} - {}".format(np.min(patches_imgs_test), np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs, test_masks


# data consinstency check
def data_consistency_check(imgs, masks):
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[2] == masks.shape[2])
    assert (imgs.shape[3] == masks.shape[3])
    assert (masks.shape[1] == 1)
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3)


# extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    if N_patches % full_imgs.shape[0] != 0:
        print("N_patches: please enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)  # masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
    print("patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
            # print "y_center " +str(y_center)
            # check whether the patch is fully contained in the FOV
            if inside and not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                continue
            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patch_mask = full_masks[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                         x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1  # total
            k += 1  # per full_img
    return patches, patches_masks


# check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to
    # contain the full patch in the FOV
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    if radius < R_inside:
        return True
    else:
        return False


# Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays (#pic, 1, height, width)
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h, img_w = full_imgs.shape[2:]  # height & width of the full image

    N_patches_h = math.ceil(img_h / patch_h)
    N_patches_w = math.ceil(img_w / patch_w)

    print('padding the full image to be patchable')
    padded_imgs = np.zeros([full_imgs.shape[0], full_imgs.shape[1], N_patches_h * patch_h, N_patches_w * patch_w])
    padded_imgs[:, :, :img_h, :img_w] = full_imgs

    print("number of patches per image: {} x {}".format(N_patches_h, N_patches_w))

    patches = np.zeros([(N_patches_h * N_patches_w) * padded_imgs.shape[0], padded_imgs.shape[1], patch_h, patch_w])

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                st_h = h * patch_h
                st_w = w * patch_w
                patches[iter_tot, :, :, :] = padded_imgs[i, :, st_h:st_h + patch_h, st_w:st_w + patch_w]
                iter_tot += 1  # total

    assert (iter_tot == (N_patches_h * N_patches_w) * padded_imgs.shape[0])
    return patches  # (#patch, 1, 48, 48)


# Recompone the full images with the patches
def recompone(patches, patch_h, patch_w, full_img_height, full_img_width):
    assert (len(patches.shape) == 4)  # (#img x #patch_h x #patch_w, 1, patch_h, patch_w)
    assert (patches.shape[1] == 1 or patches.shape[1] == 3)  # check the channel is 1 or 3

    N_patches_h = math.ceil(full_img_height / patch_h)
    N_patches_w = math.ceil(full_img_width / patch_w)
    img_num = patches.shape[0] // (N_patches_h * N_patches_w)
    assert (patches.shape[0] % (N_patches_h * N_patches_w) == 0)

    orig_imgs = np.zeros([img_num, patches.shape[1], N_patches_h * patch_h, N_patches_w * patch_w])

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(img_num):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                st_h = h * patch_h
                st_w = w * patch_w
                orig_imgs[i, :, st_h:st_h + patch_h, st_w:st_w + patch_w] = patches[iter_tot, :, :, :]
                iter_tot += 1  # total
    assert (iter_tot == patches.shape[0])

    return orig_imgs[:, :, :full_img_height, :full_img_width]


# return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs, data_masks, original_imgs_border_masks):
    assert (data_imgs.shape == data_masks.shape == original_imgs_border_masks.shape)
    assert (len(data_imgs.shape) == 4)  # 4D arrays (#img, 1, height, width)
    assert (data_imgs.shape[1] == 1)  # check the channel is 1

    return data_imgs[original_imgs_border_masks > 0], data_masks[original_imgs_border_masks > 0]
