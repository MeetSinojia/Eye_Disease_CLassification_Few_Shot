import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


def check_image_path(directory_with_images):


  correct_filepaths = []

  
  list_of_images = os.listdir(directory_with_images) 
  for img in list_of_images:
    img_filePath = directory_with_images + img
    if cv2.imread(img_filePath) is not None:
      correct_filepaths.append(img_filePath)

  return correct_filepaths


def read_eye_images(directory_path, img_height, img_width, augmentation=False):


  datax = [] # list to hold the images
  datay = [] # list to hold the image labels

  sub_directories = os.listdir(directory_path)

  for sub_dir in sub_directories: # for each sub_directory read the images

     correct_image_filepaths = check_image_path(directory_path + '/' + sub_dir + '/')
    
     for fpath in correct_image_filepaths: # loop through the image file paths

      image = cv2.imread(fpath) # this reads image in BGR format

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      image = cv2.resize(image, (img_width, img_height))
      
      if augmentation:
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        flip_vertical = cv2.flip(image, 0)
        flip_horizontal = cv2.flip(image, 1)
        flip_both = cv2.flip(image, -1)      
            
        datax.extend((image, rotated_90, rotated_180, rotated_270, flip_vertical, flip_horizontal, flip_both))
        # add the labels to our list
        datay.extend((
          sub_dir + '_0', 
          sub_dir + '_90', 
          sub_dir + '_180', 
          sub_dir + '_270',
          sub_dir + '_FV',
          sub_dir + '_FH',
          sub_dir + '_fVH'          
          ))

      else: 
        datax.append(image)
        datay.append(sub_dir)     

  return np.array(datax), np.array(datay)

def display_images(images, labels, rows = 1, cols=1):

    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for i, (img, label) in enumerate(zip(images, labels)):
        ax.ravel()[i].imshow(img)
        ax.ravel()[i].set_title(label)
        ax.ravel()[i].set_axis_off()
    
    plt.show()

def get_mean_std(image_data_directory, img_height, img_width, batch_size=1):

  unbroken_images_filepaths = check_image_path(image_data_directory) 

  print(f"Number of images used to compute the mean and standard deviation: {len(unbroken_images_filepaths)}")  
  
  image_data = DataPreProcessing(unbroken_images_filepaths, img_height, img_width)  
  
  image_dl = DataLoader(image_data, batch_size=batch_size, shuffle=False, drop_last=True) 
  

  channels_sum, channels_squared_sum, num_batches = 0, 0, 0

  for images in image_dl:

    channels_sum += torch.mean(images * 1.0, dim=[0, 2, 3])

    channels_squared_sum += torch.mean((images * 1.0)**2, dim=[0, 2, 3])

    num_batches += 1

  mean = channels_sum / num_batches

  std = (channels_squared_sum/num_batches - mean**2)**0.5

  return mean, std

class DataPreProcessing(Dataset):

  def __init__(self, images_filepaths, img_height, img_width):
    
    self.images_filepaths = images_filepaths

    self.height = img_height

    self.width = img_width
    
  
  def __len__(self):

    return len(self.images_filepaths)


  def __getitem__(self, ix):


    # Pre-processing the image
    # ------------------------
    image_filepath = self.images_filepaths[ix] # get path to image

    image = cv2.imread(image_filepath) # image read in BGR format

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image RGB format

    image = cv2.resize(image, (self.height, self.width)) # resize images to the same size

    image = torch.from_numpy(image).float() # convert to Tensor and float data

    image = image / 255 # scale image to [0, 1]

    image = image.permute(2, 0, 1) 
          
    return image