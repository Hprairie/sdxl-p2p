import os
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET
import xmltodict
from collections import defaultdict

def parse_xml(path):
    """ Parse the xml format of PASCAL VOC """
    label = {}
    def parse_object(obj):
        name = obj['name']
        # Convert to percentage format as simpler for transformations
        ymin = int(obj['bndbox']['ymin']) - 1
        xmin = int(obj['bndbox']['xmin']) - 1
        ymax = int(obj['bndbox']['ymax']) - 1
        xmax = int(obj['bndbox']['xmax']) - 1

        x = (xmax + xmin) / label['image_size'][2] / 2
        y = (ymax + ymin) / label['image_size'][1] / 2
        width = (xmax - xmin) / label['image_size'][2] 
        height = (ymax - ymin) / label['image_size'][1]

        return [name, x, y, width, height]

    # Parse XML
    with open(path, 'rb') as file:
        xml_data = xmltodict.parse(file, xml_attribs=False)

    # Get Size
    size = xml_data['annotation']['size']
    w = int(size['width'])
    h = int(size['height'])
    c = int(size['depth'])

    label['image_size'] = [c, h, w]

    bounding_boxes = []
    if isinstance(xml_data['annotation']['object'], list):
        for obj in xml_data['annotation']['object']:
            bounding_boxes.append(parse_object(obj))
    else:
        bounding_boxes.append(parse_object(xml_data['annotation']['object']))

    label['bounding_boxes'] = bounding_boxes
    return label 

def parse_txt(path):
    """ Parse the xml format of PASCAL VOC """
    label = {}
    def parse_object(line):
        name, x, y, width, height = line.split()
        return [int(name), float(x), float(y), float(width), float(height)]

    bounding_boxes = []
    # Parse TXT 
    with open(path, 'rb') as file:
        for line in file:
            bounding_boxes.append(parse_object(line))

    label['bounding_boxes'] = bounding_boxes
    return label 

def parse_xml_objects(path):
    # Parse XML
    with open(path, 'rb') as file:
        xml_data = xmltodict.parse(file, xml_attribs=False)
    objects = defaultdict(int)
    if isinstance(xml_data['annotation']['object'], list):
        for obj in xml_data['annotation']['object']:
            objects[obj['name']] += 1
    else:
        obj = xml_data['annotation']['object']
        objects[obj['name']] += 1
    return objects

def parse_txt_objects(path, decoder):
    # Parse TXT 
    objects = defaultdict(int)
    with open(path, 'r') as file:
        for line in file:
            name, _, _, _, _ = line.split()
            print(name, line)
            objects[decoder[int(name)]] += 1
    return objects

def parse_json(path):
    """ Parse the json format of MS COCO """
    pass

def process_label(label_path, ext):
    """ Coverts a label with a given extension to all relevant information for processing """
    assert ext in ['xml', 'json', 'txt'], "Ensure correct label format, Please check yaml mask configuration"

    if ext == 'xml':
        return parse_xml(label_path)
    elif ext == 'json':
        return parse_json(label_path)
    elif ext == 'txt':
        return parse_txt(label_path)

def get_dataset(path):
    """Get of sample names of all files in specified path"""
    samples = os.listdir(path)
    for i, sample in enumerate(samples):
        samples[i] = os.path.splitext(sample)[0]
    return samples

def get_sample(name, data_config):
    """Get a sample and the corresponding path to its annotations"""
    image = plt.imread(data_config['image_path'] + '/' + name + '.jpg')
    ann_path = data_config['label_path'] + '/' + name + '.' + data_config['ext']
    return image, ann_path 
