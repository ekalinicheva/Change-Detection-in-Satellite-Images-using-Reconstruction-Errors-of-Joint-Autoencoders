import os, time, re, datetime
import numpy as np
from random import sample

import torch
from torch import nn
from torch.autograd import Variable

from osgeo import gdal, ogr
import scipy.ndimage.filters as fi


from models.AE_fully_convolutional_model import Encoder, Decoder    #we chose fully_conv or conv model
from codes.imgtotensor_patches_samples_list import ImageDataset
from codes.image_processing import extend, open_tiff
from codes.loader import dsloader
from codes.check_gpu import on_gpu
from codes.plot_loss import plotting


#gaussin filter if we want to apply weight to the patch loss. Center das higher weight
def gkern2(kernlen, sigma):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, sigma)*100


#create new directory if does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


gpu = on_gpu()
print("ON GPU is "+str(gpu))



#Parameters
patch_size = 5
bands_to_keep = 3   # 4 if we keep SWIR band for SPOT or Blue band for Sentinel. Otherwise, if wee keep 3 bands, it's G, R, NIR
epoch_nb = 5
batch_size = 150
learning_rate = 0.0005
weighted = False    # if we weight patches loss (center pixel has higher loss)
sigma = 2           # sigma for weighted loss
shuffle = True      # shuffle patches before training
satellite = "SPOT5" # ["SPOT5", "S2"]


start_time = time.clock()
run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))
print (run_name)


# We define the input and output paths
folder_results = "All_images_ep_" + str(epoch_nb) + "_patch_" + str(patch_size) +"_fc"+ run_name
path_results = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/NN_Montpellier_'+str(satellite)+'_all_images_model_pretrained/') + folder_results + "/"
path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
create_dir(path_results)
path_model = path_results + 'model'+run_name+"/" #we will save the pretrained encoder/decoder models here
create_dir(path_model)


# We open all the images of time series images and mirror the borders.
# Then we create 4D array with all the images of the dataset
images_list = os.listdir(path_datasets)
path_list = []
list_image_extended= []
for image_name_with_extention in images_list:
    if image_name_with_extention.endswith(".TIF") and not image_name_with_extention.endswith("band.TIF"):
        img_path = path_datasets + image_name_with_extention
        path_list.append(img_path)
        image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
        # We keep only essential bands if needed
        if bands_to_keep==3:
            if satellite == "SPOT5":
                if bands_nb==4:
                    image_array = np.delete(image_array, 3, axis=0)
                    bands_nb = 3
                if bands_nb==8:
                    image_array = np.delete(image_array, [3,7], axis=0)
                    bands_nb = 6
            if satellite == "S2":
                image_array = np.delete(image_array, 0, axis=0)
        image_extended = extend(image_array, patch_size)    # We mirror the border rows and cols
        list_image_extended.append(image_extended)
list_image_extended = np.asarray(list_image_extended, dtype=float)


# We normalize all the images from 0 to 1 (1 is the max value of the whole dataset, not an individual image)
list_norm = []
for band in range(len(list_image_extended[0])):
    all_images_band = list_image_extended[:, band, :, :].flatten()
    min = np.min(all_images_band)
    max = np.max(all_images_band)
    list_norm.append([min, max])
for i in range(len(list_image_extended)):
    for band in range(len(list_image_extended[0])):
        list_image_extended[i][band] = (list_image_extended[i][band]-list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])


driver_tiff = gdal.GetDriverByName("GTiff")
driver_shp = ogr.GetDriverByName("ESRI Shapefile")


# We create a training dataset with patches
image = None    # Dataset with the sample of patches from all images
nbr_patches_per_image = int(H*W/len(list_image_extended))   # We sample H*W/ number of images patches from every image
# We create a dateset for every image separately and then we concatenate them
for ii in range(len(list_image_extended)):
    samples_list = np.sort(sample(range(H * W), nbr_patches_per_image))
    if image is None:
        image = ImageDataset(list_image_extended[ii], patch_size, ii,
                             samples_list)  # we create a dataset with tensor patches
    else:
        image2 = ImageDataset(list_image_extended[ii], patch_size, ii,
                              samples_list)  # we create a dataset with tensor patches
        image = torch.utils.data.ConcatDataset([image, image2])
loader = dsloader(image, gpu, batch_size, shuffle=True) # dataloader
image = None


list_image_extended = None


# We write stats to file
with open(path_results+"stats.txt", 'a') as f:
    f.write("Relu activations for every layer except the last one. L2" + "\n")
    if weighted:
        f.write("sigma=" + str(sigma) + "\n")
    else:
        f.write("Loss not weighted" + "\n")
    f.write("patch_size=" + str(patch_size) + "\n")
    f.write("epoch_nb=" + str(epoch_nb) + "\n")
    f.write("batch_size=" + str(batch_size) + "\n")
    f.write("learning_rate=" + str(learning_rate) + "\n")
    f.write("bands_to_keep= " + str(bands_to_keep) + "\n")
    f.write("Nbr patches per image " + str(nbr_patches_per_image) + "\n")
f.close()




# We create AE model
encoder = Encoder(bands_nb, patch_size) # On CPU
decoder = Decoder(bands_nb, patch_size) # On CPU
if gpu:
    encoder = encoder.to('cuda:0')  # On GPU
    decoder = decoder.to('cuda:0')  # On GPU


optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)    #optimizer
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.MSELoss()    # loss function


# We write the encoder model to stats
with open(path_results+"stats.txt", 'a') as f:
    f.write(str(encoder) + "\n")
f.close()

# calculate the weights if the patch loss is weighted
weight = torch.from_numpy(gkern2(patch_size, sigma)).float().expand(batch_size, bands_nb, patch_size, patch_size)
if gpu:
    weight = weight.cuda()

start_time = time.clock()


# Function to pretrain the model, pretty much standart
epoch_loss_list = []
def train(epoch):
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch_idx, (data, _, _) in enumerate(loader):
        if gpu:
            data = data.cuda(async=True)
        encoded = encoder(Variable(data))
        decoded = decoder(encoded)

        # we calculate batch loss to optimize the model
        if weighted:
            loss = criterion(decoded*Variable(weight), Variable(data)*Variable(weight))
        else:
            loss = criterion(decoded, Variable(data))

        total_loss += loss.item()
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        if (batch_idx+1) % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                100. * (batch_idx+1) / len(loader), loss.item()))
    epoch_loss = total_loss / len(loader)
    epoch_loss_list.append(epoch_loss)
    epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch + 1, epoch_loss)
    print(epoch_stats)
    with open(path_results + "stats.txt", 'a') as f:
        f.write(epoch_stats+"\n")
    f.close()
    # we save the model
    torch.save([encoder, decoder], (path_model+'ae-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss, 7))+run_name+'.pkl') )

    # We plot the loss values
    if (epoch+1) % 5 == 0:
        plotting(epoch+1, epoch_loss_list, path_results)


for epoch in range(epoch_nb):
    train(epoch)


# Some stats about the best epoch loss and learning time
best_epoch = np.argmin(np.asarray(epoch_loss_list))+1
best_epoch_loss = epoch_loss_list[best_epoch-1]


print("best epoch " + str(best_epoch))
print("best epoch loss " + str(best_epoch_loss))


end_time = time.clock()

total_time_learning = end_time - start_time
total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
print("Total time learning =", total_time_learning)


with open(path_results+"stats.txt", 'a') as f:
    f.write("best epoch " + str(best_epoch) + "\n")
    f.write("best epoch loss " + str(best_epoch_loss) + "\n")
    f.write("Total time learning=" + str(total_time_learning) + "\n"+"\n")
f.close()