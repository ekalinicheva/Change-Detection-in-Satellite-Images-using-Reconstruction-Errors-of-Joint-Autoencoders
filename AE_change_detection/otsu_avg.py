from skimage import filters
from osgeo import gdal
import numpy as np
from codes.image_processing import create_tiff, vectorize_tiff, open_tiff
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score


# This code is taken from skimage
def histogram(image, nbins=256):
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        offset = 0
        image_min = np.min(image)
        if image_min < 0:
            offset = image_min
            image_range = np.max(image).astype(np.int64) - image_min
            # get smallest dtype that can hold both minimum and offset maximum
            offset_dtype = np.promote_types(np.min_scalar_type(image_range),
                                            np.min_scalar_type(image_min))
            if image.dtype != offset_dtype:
                # prevent overflow errors when offsetting
                image = image.astype(offset_dtype)
            image = image - offset
        hist = np.bincount(image)
        bin_centers = np.arange(len(hist)) + offset

        # clip histogram to start with a non-zero bin
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(image.flat, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers


# This code is taken from skimage
def threshold_otsu(image, nbins=256):
    hist, bin_centers = histogram(image, nbins)
    hist = hist.astype(float)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


# Function to calculae Otsu threshold from the average of 2 reconstruction error images
def otsu(image_array_loss1, image_array_loss2, H, W, geo, proj, path_results, images_date, changes=None):
    # We calculate the average reconstruction error image
    image_array_loss = np.divide((image_array_loss1+image_array_loss2), 2)

    # We rescale the image values to 8 bits so it works with the functions from skimage
    max_ = np.max(image_array_loss)
    coef = max_/256
    image_array_loss = image_array_loss/coef
    image_array_loss = np.asarray(image_array_loss, dtype=int)

    # THIS IS VERY IMPORTANT VALUE
    # Otsu threshold is automatic, however before applying it, we exclude 0.5% of the highest reconstruction error values as they ae considered to be outliers
    # This parameter can be modified if needed
    threshold = 0.995
    val = filters.threshold_otsu(np.sort(image_array_loss.flatten())[0:int(H*W*threshold)]) # Obtained threshold value

    # We get binary change map (1 - changes, 0 - no changes) using the threshold and write it to tiff and shp
    image_array_outliers = np.zeros(H*W)
    image_array_outliers[image_array_loss.flatten() > val] = 1
    outliers_image_mean = "Outliers_average_" + images_date + "_" +str(threshold)
    dst_ds = create_tiff(1, path_results+ "/"+outliers_image_mean + ".TIF", W, H, gdal.GDT_Int16, np.reshape(image_array_outliers, (H, W)), geo, proj)
    vectorize_tiff(path_results, "/"+outliers_image_mean, dst_ds)
    dst_ds = None


    # We calculate the stats if the ground truth is available for this couple of images
    if changes is not None:
        # path of ground truth image, I have only 2 GT
        path_cm = '/home/user/Dropbox/IJCNN/images/' + changes
        cm_truth_name = "mask_changes_small1"
        if changes=="changes_2004_2005":
            cm_predicted = (np.reshape(image_array_outliers, (H, W))[0:600, 600:1400]).flatten()
        if changes == "changes_2006_2008":
            cm_predicted = (np.reshape(image_array_outliers, (H, W))[100:370, 1000:1320]).flatten()

        cm_truth, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name)
        cm_truth = cm_truth.flatten()
        cm_truth[cm_truth==255]=0
        #Different stats taken from scikit
        print(classification_report(cm_truth, cm_predicted, target_names=["no changes", "changes"]))
        print(accuracy_score(cm_truth, cm_predicted))
        print(cohen_kappa_score(cm_truth, cm_predicted))
        conf = confusion_matrix(cm_truth, cm_predicted)
        print(confusion_matrix(cm_truth, cm_predicted))
        omission = conf[1][0]/sum(conf[1])
        print (omission)


# Function to calculae Otsu threshold for both reconstruction error images separetely and then union the results
# Better not to use this one, but possible that it can give better results for some images
def otsu_independent(image_array_loss1, image_array_loss2, H, W, geo, proj, path_results, images_date, changes=None):
    # We calculate the change map for the 1st reconstruction error image. Same principle as in otsu() function
    max_ = np.max(image_array_loss1)
    coef = max_/256
    image_array_loss1 = image_array_loss1/coef
    image_array_loss1 = np.asarray(image_array_loss1, dtype=int)
    threshold = 0.995
    val = filters.threshold_otsu(np.sort(image_array_loss1.flatten())[0:int(H*W*threshold)])
    image_array_outliers = np.zeros(H*W)
    image_array_outliers[image_array_loss1.flatten() > val] = 1

    # We calculate the change map for the 2nd reconstruction error image. Same principle as in otsu() function
    max_ = np.max(image_array_loss2)
    coef = max_/256
    image_array_loss2 = image_array_loss2/coef
    image_array_loss2 = np.asarray(image_array_loss2, dtype=int)
    threshold = 0.995
    val = filters.threshold_otsu(np.sort(image_array_loss2.flatten())[0:int(H*W*threshold)])
    image_array_outliers[image_array_loss2.flatten() > val] = 1 # we add the change pixels to the results obtained from the 1st image

    # We write tiff and shp
    outliers_image_mean = "Outliers_average_" + images_date + "_independent_" +str(threshold)
    dst_ds = create_tiff(1, path_results+ "/"+outliers_image_mean + ".TIF", W, H, gdal.GDT_Int16, np.reshape(image_array_outliers, (H, W)), geo, proj)
    vectorize_tiff(path_results, "/"+outliers_image_mean, dst_ds)
    dst_ds = None


    # We calculate the classification stats if the ground truth if available
    if changes is not None:
        path_cm = 'C:/Users/Ekaterina_the_Great/Dropbox/IJCNN/images/'+changes
        path_cm = '/home/user/Dropbox/IJCNN/images/' + changes
        cm_truth_name = "mask_changes_small1"
        print(image_array_outliers.shape)
        if changes=="changes_2004_2005":
            cm_predicted = (np.reshape(image_array_outliers, (H, W))[0:600, 600:1400]).flatten()
        if changes == "changes_2006_2008":
            cm_predicted = (np.reshape(image_array_outliers, (H, W))[100:370, 1000:1320]).flatten()

        cm_truth, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name)
        cm_truth = cm_truth.flatten()
        cm_truth[cm_truth==255]=0
        #Different stats taken from scikit
        print(classification_report(cm_truth, cm_predicted, target_names=["no changes", "changes"]))
        print(accuracy_score(cm_truth, cm_predicted))
        print(cohen_kappa_score(cm_truth, cm_predicted))
        conf = confusion_matrix(cm_truth, cm_predicted)
        print(confusion_matrix(cm_truth, cm_predicted))
        omission = conf[1][0]/sum(conf[1])
        print (omission)
