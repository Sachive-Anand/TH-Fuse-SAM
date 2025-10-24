import torch
import numpy as np
import cv2
from torchvision import transforms
from imageio import imread
import os

# ✅ THIS IS THE CORRECTED FUNCTION THAT SEARCHES ALL SUBFOLDERS
def list_images(path):
    """
    Recursively finds all image paths in the 'visible' subdirectories of a given path.
    Returns a list of full, absolute paths to the visible images.
    """
    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    # os.walk will go through every folder and file in the directory tree
    for root, dirs, files in os.walk(path):
        # We only care about images inside a folder named 'visible'
        if 'visible' in root:
            for file in files:
                if file.lower().endswith(valid_extensions):
                    # Add the full path to the image
                    image_paths.append(os.path.join(root, file))
                    
    return sorted(image_paths)

# --- The rest of the functions are unchanged ---

def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, pilmode="L")
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if height is None and width is None:
            w, h = image.shape[0], image.shape[1]
            w_pad = (256 - w % 256) % 256
            h_pad = (256 - h % 256) % 256
            image = cv2.copyMakeBorder(image, 0, w_pad, 0, h_pad, cv2.BORDER_CONSTANT, value=128)
        
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    cv2.imwrite(path, ori)
