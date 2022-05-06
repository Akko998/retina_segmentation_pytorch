###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################

import configparser
import sys, os
from sys import stdout
from models.u_net import UNet
from help_functions import *
from extract_patches import get_data_training
import torch
import shutil
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
print(torch.version.cuda)

# ========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
# patch to the datasets
path_data = config.get('data paths', 'path_local')
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train, patches_imgs_vali, patches_masks_vali = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV')
    # select the patches only inside the FOV  (default == True)
)
stdout.flush()

train_set = TensorDataset(torch.tensor(patches_imgs_train), torch.tensor(patches_masks_train))
train_loader = DataLoader(train_set, batch_size, shuffle=True)

vali_set = TensorDataset(torch.tensor(patches_imgs_vali), torch.tensor(patches_masks_vali))
vali_loader = DataLoader(vali_set, batch_size, shuffle=False)

# ========= Save a sample of what you're feeding to the neural network ==========
name_experiment = config.get('experiment name', 'name')
if not os.path.exists(name_experiment):
    os.makedirs(name_experiment)

N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[:N_sample, :, :, :], 5),
          os.path.join(name_experiment, 'sample_input_imgs'))  # .show()
visualize(group_images(patches_masks_train[:N_sample, :, :, :], 5),
          os.path.join(name_experiment, 'sample_input_masks'))  # .show()

# =========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]

unet = UNet(n_ch, num_classes=1).to(device, dtype=torch.float)  # the U-net model
epoch_num = int(config.get('training settings', 'N_epochs'))
optimizer = torch.optim.Adam(unet.parameters())
loss_fn = torch.nn.BCELoss()

save_point_file = os.path.join(name_experiment, config.get('experiment name', 'model_checkpoint'))
model_best_file = os.path.join(name_experiment, config.get('experiment name', 'model_best'))

# load save point if exists
if os.path.exists(save_point_file):
    checkpoint_params = torch.load(save_point_file)

    start_epoch = checkpoint_params['epoch']
    train_loss_list = checkpoint_params['train_loss_list']
    vali_loss_list = checkpoint_params['vali_loss_list']
    min_vali_loss = checkpoint_params['min_vali_loss']
    unet.load_state_dict(checkpoint_params['model'])
    optimizer.load_state_dict(checkpoint_params['optimizer'])

    print('checkpoint detected, epoch: %d' % start_epoch)
else:
    train_loss_list, vali_loss_list = [], []
    start_epoch, min_vali_loss = 0, np.Inf

# start training
print('start training')
stdout.flush()

for epoch in range(start_epoch, epoch_num):
    train_loss, vali_loss = 0, 0

    # training
    for inputs, labels in train_loader:
        # (patch,1,width,height)
        loss = loss_fn(unet(inputs.to(device, dtype=torch.float)), labels.to(device, dtype=torch.float))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().numpy() * inputs.shape[0]
    train_loss /= patches_imgs_train.shape[0]

    # validation
    with torch.no_grad():
        for inputs, labels in vali_loader:
            loss = loss_fn(unet(inputs.to(device, dtype=torch.float)), labels.to(device, dtype=torch.float))
            vali_loss += loss.detach().cpu().numpy() * inputs.shape[0]

    vali_loss /= patches_imgs_vali.shape[0]

    # save check point & best model in each epoch
    is_best = vali_loss < min_vali_loss
    min_vali_loss = min(vali_loss, min_vali_loss)

    train_loss_list.append(train_loss)
    vali_loss_list.append(vali_loss)

    torch.save({
        'epoch': epoch + 1,
        'train_loss_list': train_loss_list,
        'vali_loss_list': vali_loss_list,
        'min_vali_loss': min_vali_loss,
        'model': unet.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_point_file)
    if is_best:
        shutil.copyfile(save_point_file, model_best_file)

    if epoch % 1 == 0:
        print("epoch %d:\t train loss %f\t test loss %f" % (epoch, train_loss, vali_loss))
        stdout.flush()

print("ploting curves")
plotCurve(range(1, epoch_num + 1), train_loss_list, os.path.join(name_experiment, 'plot_curves.svg'),
          x2_vals=range(1, epoch_num + 1), y2_vals=vali_loss_list, legend=["train", "vali"])
