import os, time, re, datetime
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from osgeo import gdal, ogr
from codes.image_processing import open_tiff, extend, create_tiff
from codes.imgtotensor_patches_samples_two import ImageDataset
from codes.loader import dsloader
from codes.check_gpu import on_gpu

from random import sample
from otsu_avg import otsu
import scipy.ndimage.filters as fi


#gaussin filter if we want to apply weight to the patch loss. Center das higher weight
def gkern2(kernlen, sigma):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, sigma)*100


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


gpu = on_gpu()  #We check if we can work on GPU, GPU RAM should be >4Gb
print ("ON GPU is "+str(gpu))


start_time = time.clock()
run_name = str(time.strftime(".%Y-%m-%d_%H%M"))
print (run_name)


#Here are the names of the couples of images
# #10m SPOT-5 image
image_name1 = 'SPOT5_HRG1_XS_20021005_N1_SCENE_047_262'
image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
image_name2 = 'SPOT5_HRG1_XS_20030918_N1_SCENE_047_262_0'
image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
#
# image_name1 = 'SPOT5_HRG1_XS_20030918_N1_SCENE_047_262_0'
# image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# image_name2 = 'SPOT5_HRG1_XS_20040514_N1_SCENE_047_262'
# image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # # #
# # # #
# image_name1 = 'SPOT5_HRG1_XS_20040514_N1_SCENE_047_262'
# image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# image_name2 = 'SPOT5_HRG2_XS_20040822_N1_SCENE_047_262'
# image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# image_name1 = 'SPOT5_HRG2_XS_20040822_N1_SCENE_047_262'
# image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# image_name2 = 'SPOT5_HRG2_XS_20050427_N1_SCENE_047_262_0'
# image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# # image_name1 = 'SPOT5_HRG2_XS_20050427_N1_SCENE_047_262_0'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG1_XS_20051201_N1_SCENE_047_262_0'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# #
# # image_name1 = 'SPOT5_HRG1_XS_20051201_N1_SCENE_047_262_0'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20060218_N1_SCENE_047_261_9'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# #
# # image_name1 = 'SPOT5_HRG2_XS_20060218_N1_SCENE_047_261_9'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20060603_N1_SCENE_047_262_0'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# #
# # image_name1 = 'SPOT5_HRG2_XS_20060603_N1_SCENE_047_262_0'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20070201_N1_SCENE_047_262_0'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# #
# # image_name1 = 'SPOT5_HRG2_XS_20070201_N1_SCENE_047_262_0'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20070406_N1_SCENE_047_262'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# # #
# # image_name1 = 'SPOT5_HRG2_XS_20070406_N1_SCENE_047_262'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20080621_N1_SCENE_046_261_9'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# # image_name1 = 'SPOT5_HRG2_XS_20080621_N1_SCENE_046_261_9'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20080821_N1_SCENE_047_262'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# # image_name1 = 'SPOT5_HRG2_XS_20060218_N1_SCENE_047_261_9'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20070201_N1_SCENE_047_262_0'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # #
# # image_name1 = 'SPOT5_HRG2_XS_20070201_N1_SCENE_047_262_0'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20080621_N1_SCENE_046_261_9'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# # # #
# # image_name1 = 'SPOT5_HRG2_XS_20060218_N1_SCENE_047_261_9'
# # image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# # image_name2 = 'SPOT5_HRG2_XS_20080821_N1_SCENE_047_262'
# # image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)
# # #
# #
# image_name1 = 'SPOT5_HRG1_XS_20040514_N1_SCENE_047_262'
# image_date1 = (re.search("_([0-9]*)_", image_name1)).group(1)
# image_name2 = 'SPOT5_HRG2_XS_20050427_N1_SCENE_047_262_0'
# image_date2 = (re.search("_([0-9]*)_", image_name2)).group(1)


#Parameters

#The parameters for the model
patch_size = 5
bands_to_keep = 3   # 4 if we keep SWIR band for SPOT or Blue band for Sentinel. Otherwise, if wee keep 3 bands, it's G, R, NIR
epoch_nb = 1
batch_size = 150
learning_rate = 0.00005
weighted = False    # if we weight patches loss (center pixel has higher loss)
sigma = 2           # sigma for weighted loss
shuffle = True      # shuffle patches before training
sampleTrue = False  # if we train the model with all the patches or only with samles
satellite = "SPOT5" # ["SPOT5", "S2"]


# Path for pretrained model
reference_model = "2019-04-23_1505" # unique "time" of the model
epoch_model = 5     # best epoch
loss_model = 2.27e-05   #loss of the best epoch (taken from the .pkl file)
path_models = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/OUTLIERS/NN_Montpellier_SPOT5_all_images_model_pretrained/') # Path for all pretrained models
folder_pretrained_results = "All_images_ep_5_patch_5_fc.2019-04-23_1505/" #folder for the concrete model


# Input and output data paths
path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
folder_results = folder_pretrained_results + "Joint_AE_"+image_date1 + "_" +image_date2 + "_ep_" + str(epoch_nb) + "_patch_" + str(patch_size) + run_name
path_results = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/NN_Montpellier_'+str(satellite)+'_all_images_model_pretrained_/') + folder_results+"/"
create_dir(path_results)
path_model_finetuned = path_results + 'model'+run_name+"/" #we will save the finutuned encoder/decoder models here
create_dir(path_model_finetuned)



# We load the model
model_folder = folder_pretrained_results + "model."+reference_model
ae_model_name = "ae-model_ep_"+str(epoch_model) + "_loss_"+str(loss_model)+"."+reference_model
encoder12, decoder12 = torch.load(path_models+model_folder+"/"+ae_model_name+".pkl")
encoder21, decoder21 = torch.load(path_models+model_folder+"/"+ae_model_name+".pkl")


if gpu:
    encoder12 = encoder12.cuda()  # On GPU
    decoder12 = decoder12.cuda()  # On GPU
    encoder21 = encoder21.cuda()  # On GPU
    decoder21 = decoder21.cuda()  # On GPU


driver_tiff = gdal.GetDriverByName("GTiff")
driver_shp = ogr.GetDriverByName("ESRI Shapefile")


#We open a couple of images to detect changes
image_array1, H, W, geo, proj, bands_nb = open_tiff(path_datasets, image_name1)
image_array2, H, W, geo, proj, bands_nb = open_tiff(path_datasets, image_name2)
if bands_to_keep == 3:
    if satellite == "SPOT5":
        if bands_nb==8:
            image_array1 = np.delete(image_array1, [3, 7], axis=0)
            image_array2 = np.delete(image_array2, [3, 7], axis=0)
            bands_nb = 6
        if bands_nb==4:
            image_array1 = np.delete(image_array1, 3, axis=0)
            image_array2 = np.delete(image_array2, 3, axis=0)
            bands_nb = 3
    if satellite == "S2":
        image_array1 = np.delete(image_array1, 0, axis=0)
        image_array2 = np.delete(image_array2, 0, axis=0)
#We mirror the border rows and cols
image_extended1 = extend(image_array1, patch_size).astype(float)
image_extended2 = extend(image_array2, patch_size).astype(float)


#We open extended images just to get the max and min values for the normalization of the current couple of images
images_list = os.listdir(path_datasets)
path_list = []
list_image_extended = []
for image_name_with_extention in images_list:
    if image_name_with_extention.endswith(".TIF") and not image_name_with_extention.endswith("band.TIF"):
        img_path = path_datasets + image_name_with_extention
        path_list.append(img_path)
        image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
        # We keep only essential bands if needed
        if bands_to_keep==3:
            if satellite == "SPOT5":
                image_array = np.delete(image_array, 3, axis=0)
                bands_nb = 3
            if satellite == "S2":
                image_array = np.delete(image_array, 0, axis=0)
        image_extended = extend(image_array, patch_size)
        list_image_extended.append(image_extended)
list_image_extended = np.asarray(list_image_extended, dtype=float)



# We calculate min and max of dataset to perform the normalization from 0 to 1 (1 is the max value of the whole dataset, not an individual image)
list_norm = []
for band in range(len(list_image_extended[0])):
    all_images_band = list_image_extended[:, band, :, :].flatten()
    min = np.min(all_images_band)
    max = np.max(all_images_band)
    list_norm.append([min, max])


# We normalize the couple of images
for band in range(len(image_extended1)):
    image_extended1[band] = (image_extended1[band] - list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])
    image_extended2[band] = (image_extended2[band] - list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])

# Create the dataset for finetuning
# If we finetune the images only on the sample of patches
if sampleTrue:
    nbr_patches_per_image = int(H * W / 2)
    samples_list = np.sort(sample(range(H * W), nbr_patches_per_image))
    image = ImageDataset(image_extended1, image_extended2, patch_size,
                         samples_list)  # we create a dataset with tensor patches
    loader = dsloader(image, gpu, batch_size, shuffle)
else:
    image = ImageDataset(image_extended1, image_extended2, patch_size,
                         list(range(H * W)))  # we create a dataset with tensor patches
    loader = dsloader(image, gpu, batch_size, shuffle)


# Create the dataset for the encoding
image_enc = ImageDataset(image_extended1, image_extended2, patch_size,
                         list(range(H * W)))  # we create a dataset with tensor patches


#we save everything to stats file
with open(path_results+"stats.txt", 'a') as f:
    f.write("Relu activations for every layer except the last one. The last one is not activated" + "\n")
    f.write("Two two sides optimized" + "\n")
    if weighted:
        f.write("sigma=" + str(sigma) + "\n")
    else:
        f.write("Loss not weighted" + "\n")
    f.write("patch_size=" + str(patch_size) + "\n")
    f.write("epoch_nb=" + str(epoch_nb) + "\n")
    f.write("batch_size=" + str(batch_size) + "\n")
    f.write("learning_rate=" + str(learning_rate) + "\n")
    f.write("sample=" + str(sampleTrue) + "\n")
f.close()


# Model optimizer
optimizer = torch.optim.Adam((list(encoder12.parameters()) + list(decoder12.parameters()) + list(encoder21.parameters()) + list(decoder21.parameters())), lr=learning_rate)
criterion = nn.MSELoss()    #loss function

# We write the encoder model to stats
with open(path_results+"stats.txt", 'a') as f:
    f.write(str(encoder12) + "\n")
f.close()


weight = torch.from_numpy(gkern2(patch_size, sigma)).float().expand(batch_size, bands_nb, patch_size, patch_size)
if gpu:
    weight = weight.cuda()

start_time = time.clock()


#function to train the model
epoch_loss_list = []
epoch_loss12_list = []
epoch_loss21_list = []
def train(epoch):
    encoder12.train() #we swich to train mode (by default)
    decoder12.train()
    encoder21.train() #we swich to train mode (by default)
    decoder21.train()
    total_loss = 0
    total_loss12 = 0
    total_loss21 = 0
    for batch_idx, (data1, data2, _) in enumerate(loader):  #we load batches from model
        if gpu:
            data1 = data1.cuda(async=True)
            data2 = data2.cuda(async=True)

        encoded12 = encoder12(Variable(data1))
        encoded21 = encoder21(Variable(data2))
        decoded12 = decoder12(encoded12)
        decoded21 = decoder21(encoded21)


        # We calculate the loss of the bottleneck
        encoded21_copy = encoded21.clone().detach()
        encoded12_copy = encoded12.clone().detach()
        loss11 = criterion(encoded12, (encoded12_copy+encoded21_copy)/2)
        loss22 = criterion(encoded21, (encoded12_copy+encoded21_copy)/2)

        # We calculate the reconstruction loss for two AEs
        total_loss += loss11.item()                 #total loss for the epoch
        loss12 = criterion(decoded12, Variable(data2))
        total_loss12 += loss12.item()
        loss21 = criterion(decoded21, Variable(data1))
        total_loss21 += loss21.item()
        optimizer.zero_grad()               #everything to optimize the model
        loss11.backward(retain_graph=True)  # retain_graph is used to deal with complex model with many branches
        loss22.backward(retain_graph=True)
        loss12.backward(retain_graph=True)
        loss21.backward()
        optimizer.step()
    # all the stats
        if (batch_idx+1) % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}\tLoss12: {:.7f}\tLoss21: {:.7f}'.format(
                (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                100. * (batch_idx+1) / len(loader), loss11.item(), loss12.item(), loss21.item()))
    epoch_loss = total_loss / len(loader)   #avg loss per epoch
    epoch_loss_list.append(epoch_loss)
    epoch_loss12 = total_loss12 / len(loader)   #avg loss per epoch
    epoch_loss12_list.append(epoch_loss12)
    epoch_loss21 = total_loss21 / len(loader)   #avg loss per epoch
    epoch_loss21_list.append(epoch_loss21)
    epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}\tAvg. Loss12: {:.7f}\tAvg. Loss21: {:.7f}".format(epoch + 1, epoch_loss, epoch_loss12, epoch_loss21)
    print(epoch_stats)
    with open(path_results + "stats.txt", 'a') as f:
        f.write(epoch_stats+"\n")
    f.close()

    #we save all the models to choose the best afterwards
    torch.save([encoder12, decoder12], (path_model_finetuned+'ae12-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss12, 7))+run_name+'.pkl') )
    torch.save([encoder21, decoder21], (path_model_finetuned+'ae21-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss21, 7))+run_name+'.pkl') )





for epoch in range(epoch_nb):
    #learning rate decay
    if epoch==3:
        learning_rate = 0.00005
        optimizer = torch.optim.Adam((list(encoder12.parameters()) + list(decoder12.parameters()) + list(encoder21.parameters()) + list(decoder21.parameters())), lr=learning_rate)
        with open(path_results + "stats.txt", 'a') as f:
            f.write("new_learning_rate=" + str(learning_rate) + "\n")
        f.close()
    train(epoch)



end_time = time.clock()

total_time_learning = end_time - start_time
total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
print("Total time learning =", total_time_learning)

with open(path_results+"stats.txt", 'a') as f:
    f.write("Total time learning=" + str(total_time_learning) + "\n"+"\n")
f.close()

# We create a datasetloader to encode the images
batch_size = H
loader = dsloader(image_enc, gpu, batch_size, shuffle=False)
criterion = nn.MSELoss(reduce=False)

# weight = torch.from_numpy(gkern2(patch_size, sigma)).float().expand(batch_size, bands_nb, patch_size, patch_size)
weight = torch.from_numpy(gkern2(patch_size, sigma)).float()
if gpu:
    weight = weight.cuda()


# We encode-decode the images and calculate the reconstruction error for the model obtained after every epoch
for best_epoch in range(1, epoch_nb+1):
    # We load the model
    best_epoch_loss12 = epoch_loss12_list[best_epoch-1]
    best_epoch_loss21 = epoch_loss21_list[best_epoch-1]
    best_encoder12, best_decoder12 = torch.load(path_model_finetuned + 'ae12-model_ep_' + str(best_epoch) + "_loss_" + str(round(best_epoch_loss12, 7)) + run_name + '.pkl')
    best_encoder21, best_decoder21 = torch.load(path_model_finetuned + 'ae21-model_ep_' + str(best_epoch) + "_loss_" + str(round(best_epoch_loss21, 7)) + run_name + '.pkl')


    if gpu:
        best_encoder12 = best_encoder12.cuda()  # On GPU
        best_encoder21 = best_encoder21.cuda()  # On GPU
        best_decoder12 = best_decoder12.cuda()  # On GPU
        best_decoder21 = best_decoder21.cuda()  # On GPU

    name_results12 = "From_" + image_date1 + "_to_" + image_date2 + "_ep_" + str(best_epoch)
    name_results21 = "From_" + image_date2 + "_to_" + image_date1 + "_ep_" + str(best_epoch)

    # We create empty arrays for reconstruction error images and for the reconstructed images
    new_coordinates_reconstructed12 = np.empty((0, bands_nb), float)
    new_coordinates_loss_mean12 = []
    new_coordinates_reconstructed21 = np.empty((0, bands_nb), float)
    new_coordinates_loss_mean21 = []

    #We switch to the evaluation mode
    best_encoder12.eval()
    best_decoder12.eval()
    best_encoder21.eval()
    best_decoder21.eval()
    for batch_idx, (data1, data2, _) in enumerate(loader):  # we load batches from model
        if gpu:
            data1 = data1.cuda(async=True)
            data2 = data2.cuda(async=True)
        encoded12 = best_encoder12(Variable(data1))
        decoded12 = best_decoder12(encoded12)
        encoded21 = best_encoder21(Variable(data2))
        decoded21 = best_decoder21(encoded21)

        #We calculate the loss
        if weighted:
            loss12 = criterion(decoded12 * Variable(weight.expand(decoded12.size()[0], decoded12.size()[1], decoded12.size()[2], decoded12.size()[3])), Variable(data2) * Variable(weight.expand(decoded12.size()[0], decoded12.size()[1], decoded12.size()[2], decoded12.size()[3])))
            loss21 = criterion(decoded21 * Variable(weight.expand(decoded21.size()[0], decoded21.size()[1], decoded21.size()[2], decoded21.size()[3])), Variable(data1) * Variable(weight.expand(decoded21.size()[0], decoded21.size()[1], decoded21.size()[2], decoded21.size()[3])))
        else:
            loss12 = criterion(decoded12, Variable(data2))
            loss21 = criterion(decoded21, Variable(data1))

        #We transform the loss values in the array
        loss_mean12 = loss12.view(-1, bands_nb, patch_size*patch_size).mean(2).mean(1)
        loss_mean21 = loss21.view(-1, bands_nb, patch_size*patch_size).mean(2).mean(1)

        if gpu:
            new_coordinates_loss_batch_mean12 = loss_mean12.data.cpu().numpy()
            new_coordinates_batch12 = decoded12.data.cpu().numpy()
            new_coordinates_loss_batch_mean21 = loss_mean21.data.cpu().numpy()
            new_coordinates_batch21 = decoded21.data.cpu().numpy()
        else:
            new_coordinates_loss_batch_mean12 = loss_mean12.data.numpy()
            new_coordinates_batch12 = decoded12.data.numpy()
            new_coordinates_loss_batch_mean21 = loss_mean21.data.numpy()
            new_coordinates_batch21 = decoded21.data.numpy()


        new_coordinates_loss_mean12.append(list(new_coordinates_loss_batch_mean12))
        new_coordinates_reconstructed12 = np.concatenate((new_coordinates_reconstructed12, new_coordinates_batch12[:, :, int(patch_size/2), int(patch_size/2)]), axis=0)

        new_coordinates_loss_mean21.append(list(new_coordinates_loss_batch_mean21))
        new_coordinates_reconstructed21 = np.concatenate((new_coordinates_reconstructed21, new_coordinates_batch21[:, :, int(patch_size/2), int(patch_size/2)]), axis=0)

        if (batch_idx + 1) % 200 == 0:
            print('Encoding : [{}/{} ({:.0f}%)]'.format(
                (batch_idx + 1) * batch_size, len(loader.dataset),
                100. * (batch_idx + 1) / len(loader)))
    new_coordinates_loss_mean12 = np.asarray(new_coordinates_loss_mean12).flatten()
    new_coordinates_loss_mean21 = np.asarray(new_coordinates_loss_mean21).flatten()


    # We create a loss image in new coordinate system for reconstruction of 2nd image from the 1st
    image_array_loss1 = np.reshape(new_coordinates_loss_mean12, (H, W))
    loss_image_mean = path_results + "Loss_mean_" + name_results12 + ".TIF"
    dst_ds = create_tiff(1, loss_image_mean, W, H, gdal.GDT_Float32, image_array_loss1, geo, proj)
    dst_ds = None


    # We reconstruct the 2nd image from the 1st
    image_array_tr = np.reshape(new_coordinates_reconstructed12, (H, W, bands_nb))
    image_array = np.transpose(image_array_tr, (2, 0, 1))
    for b in range(len(list(image_array))):
        image_array[b] = image_array[b] * (list_norm[b][1] - list_norm[b][0]) + list_norm[b][0]
    reprojected_image = path_results + "Encoded_decoded_" + name_results12 + ".TIF"
    dst_ds = create_tiff(bands_nb, reprojected_image, W, H, gdal.GDT_Int16, image_array, geo, proj)
    dst_ds = None


    # We create a loss image in new coordinate system of 1st image from the 2nd
    image_array_loss2 = np.reshape(new_coordinates_loss_mean21, (H, W))
    loss_image_mean = path_results + "Loss_mean_" + name_results21 + ".TIF"
    dst_ds = create_tiff(1, loss_image_mean, W, H, gdal.GDT_Float32, image_array_loss2, geo, proj)
    dst_ds = None


    # We reconstruct the image from the model
    image_array_tr = np.reshape(new_coordinates_reconstructed21, (H, W, bands_nb))
    image_array = np.transpose(image_array_tr, (2, 0, 1))
    for b in range(len(list(image_array))):
        image_array[b] = image_array[b] * (list_norm[b][1] - list_norm[b][0]) + list_norm[b][0]
    reprojected_image = path_results + "Encoded_decoded_" + name_results21 + ".TIF"
    dst_ds = create_tiff(bands_nb, reprojected_image, W, H, gdal.GDT_Int16, image_array, geo, proj)
    dst_ds = None

    #Here we calculate change map using otsu threshold on the average reconstruction error image
    otsu(image_array_loss1, image_array_loss2, H, W, geo, proj, path_results, image_date1 + "_to_" + image_date2, changes="changes_2004_2005")
