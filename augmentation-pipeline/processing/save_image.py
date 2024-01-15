import shutil
import os
import torch
import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET
from einops import rearrange
from processing.process_image import process_label

def save_synthetic_samples_xml(samples, label, name, data_config, skip_label, extra=''):
    # Create Directory Path if it doesn't exist
    if not os.path.exists(os.path.join(data_config['save_path'], 'labels')):
        os.makedirs(os.path.join(data_config['save_path'], 'labels'))
    if not os.path.exists(os.path.join(data_config['save_path'], 'images')):
        os.makedirs(os.path.join(data_config['save_path'], 'images'))

    # Get the label and edit its new transformation information
    xml_label = ET.parse(os.path.join(data_config['label_path'], f"{name}.{data_config['ext']}"))
    root_label = xml_label.getroot()

    # Set Path information
    # root_label.find('folder').text = os.path.join(data_config['save_path'], 'images')

    # Set Source to Synthetic
    # root_label.find('source').find('database').text = 'SYNTHETIC_IGNORE_SEGMENTATION'

    # Set Image Size
    root_label.find('size').find('depth').text = str(label['image_size'][1])
    root_label.find('size').find('height').text = str(label['image_size'][2])
    root_label.find('size').find('width').text = str(label['image_size'][3])
    
    for i, obj in enumerate(root_label.iter('object')):
        assert obj.find('name').text == label['bounding_boxes'][i][0], "Name Mismatch, Synthetic Sampling Unreliable"

        # Convert relative bounding boxes to pixel
        _, x, y, w, h = label['bounding_boxes'][i]
        _, _, H, W = label['image_size']

        x *= W
        w *= W
        y *= H
        h *= H

        xmin = int((x+1) - (w // 2))
        xmax = int((x+1) + (w // 2))
        ymin = int((y+1) - (h // 2))
        ymax = int((y+1) + (h // 2))

        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(xmin)
        bndbox.find('ymin').text = str(ymin)
        bndbox.find('xmax').text = str(xmax)
        bndbox.find('ymax').text = str(ymax)

    # Iterate through samples saving them to the path
    for i, sample in enumerate(samples):
        sample.save(os.path.join(data_config['save_path'], f"images/{name}_syn{extra}_{i+1}.jpg"), 'JPEG')

        if not skip_label:
            # Edit filepath information
            root_label.find('filename').text = f'{name}_syn{extra}_{i+1}.jpg'

            # Save Synthetic label
            xml_label.write(os.path.join(data_config['save_path'], f"labels/{name}_syn{extra}_{i+1}.{data_config['ext']}"))


def save_synthetic_samples_txt(samples, label, name, data_config, skip_label, extra=''):
    if not os.path.exists(os.path.join(data_config['save_path'], 'labels')):
        os.makedirs(os.path.join(data_config['save_path'], 'labels'))
    if not os.path.exists(os.path.join(data_config['save_path'], 'images')):
        os.makedirs(os.path.join(data_config['save_path'], 'images'))
    

    for i, sample in enumerate(samples):
        sample.save(os.path.join(data_config['save_path'], f"images/{name}_syn{extra}_{i+1}.jpg"), 'JPEG')

        if not skip_label:
            label_loc = os.path.join(data_config['label_path'], f"{name}.{data_config['ext']}")
            shutil.copy(label_loc,
                        os.path.join(data_config['save_path'], f"labels/{name}_syn{extra}_{i+1}.{data_config['ext']}"))


def save_synthetic_samples(org_image, mask, samples, skip_label, data_config, name):
    if not os.path.exists(os.path.join(data_config['save_path'], 'labels')):
        os.makedirs(os.path.join(data_config['save_path'], 'labels'))
    if not os.path.exists(os.path.join(data_config['save_path'], 'images')):
        os.makedirs(os.path.join(data_config['save_path'], 'images'))
    
    image = torch.clamp((org_image+1.0)/2.0, min=0.0, max=1.0)
    image = image.cpu().numpy()
    mask = mask.cpu().numpy()

    for i, x_sample in enumerate(samples):
        inpainted = (mask)*image+(1-mask)*x_sample.cpu().numpy()
        inpainted = inpainted.transpose(0,2,3,1)[0]*255
        Image.fromarray(inpainted.astype(np.uint8)).save(os.path.join(data_config['save_path'], f"images/{name}_syn_{i+1}.jpg"), 'JPEG')

        if not skip_label:
            label_loc = os.path.join(data_config['label_path'], f"{name}.{data_config['ext']}")
            shutil.copy(label_loc,
                        os.path.join(data_config['save_path'], f"labels/{name}_syn_{i+1}.{data_config['ext']}"))

