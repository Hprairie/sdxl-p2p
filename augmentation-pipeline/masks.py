import torch
from scipy.ndimage import distance_transform_edt
import numpy as np

# ---------------- Specific Masks Outpaint ----------------- #
def full_outpaining(label, param):
    """ Create an outpainting mask where all pixels not in the bounding box
        are masked. """
    B, C, H, W = label['image_size']
    mask = torch.zeros(label['image_size'])
    margin = param['margin']

    for box in label['bounding_boxes']:
        _, x_center, y_center, width, height = box 

        # Convert normalized coordinates to pixel coordinates
        x_center *= W
        y_center *= H 
        width *= W 
        height *= H 
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Apply the margin
        x1 = max(0, x1 - margin[0])
        y1 = max(0, y1 - margin[1])
        x2 = min(W - 1, x2 + margin[0])
        y2 = min(H - 1, y2 + margin[1])

        mask[...,y1:y2, x1:x2] = 1

    return mask

def random_outpainting(label, param):
    """ Create an outpainting mask which is random. All pixels outside of a bounding box,
        have a uniform probability of being masked. """
    B, C, H, W = label['image_size']
    # Generate Boundin Box mask
    mask = full_outpaining(label, param)

    # Create a tensor of random noise
    l, h = param['distribution']
    prob = np.random.uniform(l, h)

    # Create Random Mask
    total = H * W 
    mask_vec = np.ones([total])
    samples = np.random.choice(total, int(total * prob), replace=False)
    mask_vec[samples] = 0
    mask_vec = mask_vec.reshape(H,W)

    # Combine masks and replicate
    combined_mask = np.logical_or(mask[0,0].detach().numpy(), mask_vec)
    combined_mask = np.repeat(combined_mask[np.newaxis, np.newaxis, :, :], B * C, axis=0)

    # Reshape the combined mask to match the shape of the input
    combined_mask = combined_mask.reshape(B, C, H, W)
    combined_mask = torch.from_numpy(combined_mask).float()
    return combined_mask 


def gaussian_outpainting(label, param):
    """ Create Outpainting mask which is gaussian from bounding boxes. As the distance 
        of a pixel gets farther away from the closest bounding box pixel, then its probability
        of being masked increases. """
    B, C, H, W = label['image_size']
    # Generate Boundin Box mask
    mask = full_outpaining(label, param)

    # Compute the distance transform from the mask
    dist = distance_transform_edt(1 - mask[0, 0].detach().numpy())
    dist = dist / np.amax(dist)

    # Create gaussian mask from distance
    mu, sigma = param['distribution'] 
    rand = np.random.normal(mu, sigma, dist.shape)
    mask_dist = np.where(dist < rand, 1, 0)
    combined_mask = np.logical_or(mask[0,0].detach().numpy(), mask_dist)

    # Replicate the combined mask for all channels and batches
    combined_mask = np.repeat(combined_mask[np.newaxis, np.newaxis, :, :], B * C, axis=0)

    # Reshape the combined mask to match the shape of the input
    combined_mask = combined_mask.reshape(B, C, H, W)
    combined_mask = torch.from_numpy(combined_mask).float()
    return combined_mask 

# ---------------- Specific Masks Inpainting ----------------- #

def full_inpainting(label, param):
    """ Create an inpainting mask where all pixels in the bounding box
        are masked. """
    B, C, H, W = label['image_size']
    mask = torch.ones(label['image_size'])
    margin = param['margin']

    for box in label['bounding_boxes']:
        _, x_center, y_center, width, height = box 

        # Convert normalized coordinates to pixel coordinates
        x_center *= W
        y_center *= H 
        width *= W 
        height *= H 
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Apply the margin
        x1 = max(0, x1 - margin[0])
        y1 = max(0, y1 - margin[1])
        x2 = min(W - 1, x2 + margin[0])
        y2 = min(H - 1, y2 + margin[1])

        mask[...,y1:y2, x1:x2] = 0

    return mask

def random_inpainting(label, param):
    """ Create an inpainting mask which is random. All pixels inside of a bounding box
        have a uniform probability of being masked. """
    B, C, H, W = label['image_size']
    # Generate Boundin Box mask
    mask = full_inpainting(label, param)

    # Create a tensor of random noise
    l, h = param['distribution']
    prob = np.random.uniform(l, h)

    # Create Random Mask
    total = H * W 
    mask_vec = np.ones([total])
    samples = np.random.choice(total, int(total * prob), replace=False)
    mask_vec[samples] = 0
    mask_vec = mask_vec.reshape(H,W)

    # Combine masks and replicate
    combined_mask = np.logical_or(mask[0,0].detach().numpy(), mask_vec)
    combined_mask = np.repeat(combined_mask[np.newaxis, np.newaxis, :, :], B * C, axis=0)

    # Reshape the combined mask to match the shape of the input
    combined_mask = combined_mask.reshape(B, C, H, W)
    combined_mask = torch.from_numpy(combined_mask).float()
    return combined_mask 

# ---------------- Specific Masks Entire ----------------- #

def random_entire(label, param):
    """ Create an mask which is random. All pixels in the image 
        have a uniform probability of being masked. """
    B, C, H, W = label['image_size']

    # Create a tensor of random noise
    l, h = param['distribution']
    prob = np.random.uniform(l, h)

    # Create Random Mask
    total = H * W 
    mask_vec = np.ones([total])
    samples = np.random.choice(total, int(total * prob), replace=False)
    mask_vec[samples] = 0
    combined_mask = mask_vec.reshape(H,W)

    # Combine masks and replicate
    combined_mask = np.repeat(combined_mask[np.newaxis, np.newaxis, :, :], B * C, axis=0)

    # Reshape the combined mask to match the shape of the input
    combined_mask = combined_mask.reshape(B, C, H, W)
    combined_mask = torch.from_numpy(combined_mask).float()
    return combined_mask 

def gaussian_entire(label, param):
    """ Create Outpainting mask which is gaussian from bounding boxes. As the distance 
        of a pixel gets farther away from the closest bounding box pixel, then its probability
        of being masked increases. Additionally all pixels within the bounding box have a uniform
        probability of getting masked. """
    B, C, H, W = label['image_size']
    # Generate Bounding Box mask
    mask = full_outpaining(label, param)

    # Compute the distance transform from the mask
    dist = distance_transform_edt(1 - mask[0, 0].detach().numpy())
    dist = dist / np.amax(dist)

    # Create gaussian mask from distance
    mu, sigma = param['distribution'] 
    rand = np.random.normal(mu, sigma, dist.shape)
    mask_dist = np.where(dist < rand, 1, 0)
    combined_mask = np.logical_or(mask[0,0].detach().numpy(), mask_dist)

    # Add Random noise to the interior
    random = random_inpainting(label, param)
    combined_mask = np.logical_and(combined_mask, random[0,0].detach().numpy())

    # Replicate the combined mask for all channels and batches
    combined_mask = np.repeat(combined_mask[np.newaxis, np.newaxis, :, :], B * C, axis=0)

    # Reshape the combined mask to match the shape of the input
    combined_mask = combined_mask.reshape(B, C, H, W)
    combined_mask = torch.from_numpy(combined_mask).float()
    return combined_mask 

# ---------------- General Masks ----------------- #

def outpainting(label, param):
    assert param['method'] in ['gaussian', 'random', 'full'], "Unknown outpainting method, check mask configuration"

    if param['method'] == 'gaussian':
        return gaussian_outpainting(label, param)
    elif param['method'] == 'random':
        return random_outpainting(label, param)
    elif param['method'] == 'full':
        return full_outpaining(label, param)

def inpainting(label, param):
    assert param['method'] in ['random', 'full'], "Unknown inpainting method, check mask configuration"

    if param['method'] == 'random':
        return random_inpainting(label, param)
    elif param['method'] == 'full':
        return full_inpainting(label, param)

def entire(label, param):
    assert param['method'] in ['random', 'gaussian'], "Unknown maskind method, check mask configuration"

    if param['method'] == 'random':
        return random_entire(label, param)
    if param['method'] == 'gaussian':
        return gaussian_entire(label, param)

