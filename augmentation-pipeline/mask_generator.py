from masks import *
from processing.process_image import process_label 

class mask_generator:
    """ Class generates mask based on information within a given image. For example,
        several masks utilize bounding box information to create a mask. """
    def __init__(self, mask_type=None, mask_param=None, file_type=None, image_size=None, batch_size=None, device=None):
        self.mask_type = mask_type
        self.mask_param = mask_param 
        self.file_type = file_type
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        self.processed_label = None

    def __call__(self, label):
        assert self.mask_type in ['outpainting', 'inpainting', 'entire'], "Unkown Label, Please check yaml mask configuration"
        processed_label = process_label(label, self.file_type)

        # Override parsed label information with desired values
        if self.image_size is not None:
            processed_label['image_size'] = self.image_size
        else:
            processed_label['image_size'][1] = (processed_label['image_size'][1] + 0x3F) & (~0x3F)
            processed_label['image_size'][2] = (processed_label['image_size'][2] + 0x3F) & (~0x3F)

        if self.batch_size is not None:
            processed_label['image_size'] = [self.batch_size] + processed_label['image_size']
        else:
            processed_label['image_size'] = [1] + processed_label['image_size']

        # Save processed_label
        self.processed_label = processed_label

        if self.mask_type == 'outpainting':
            return outpainting(processed_label, self.mask_param).to(self.device)
        elif self.mask_type == 'inpainting':
            return inpainting(processed_label, self.mask_param).to(self.device)
        elif self.mask_type == 'entire':
            return entire(processed_label, self.mask_param).to(self.device)

    def get_mask_info(self):
        return self.processed_label

