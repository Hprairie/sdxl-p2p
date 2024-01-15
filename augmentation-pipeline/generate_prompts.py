import sys
sys.path.append('LLaVA/')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import CLIPProcessor, CLIPModel

model_path = "liuhaotian/llava-v1.5-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import os
import requests
import yaml
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from processing.process_image import parse_xml_objects, get_dataset, get_sample
from transformers import TextStreamer
import torch
import numpy as np

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def caption_image(image, prompt, samples):
    disable_torch_init()
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    labels = []
    for _ in range(samples):
        with torch.inference_mode():
          output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, 
                                      max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        output = outputs.rsplit('</s>', 1)[0]
        labels.append(output)
    return labels 

def get_prompt(image, obj, samples):
    text_obj = ""
    for key in obj:
        text_obj += str(obj[key]) + " " + key + ", "

    prompt = "Please describe the " + text_obj + "in this image as a caption/label, be very descriptive but concise about all aspect of only these objects including their color, location, and positon. DO NOT respond with anything other than this caption/label. DO NOT describe any other objects within the image."
    print(prompt)

    # Generate Labels from LLaVA
    generated_labels = caption_image(image, prompt, samples)

    print(f'Generate the following prompts = {generated_labels}')

    # Choose the best prompt using clip
    inputs = clip_processor(text=generated_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs) 

    logits_per_prompt = outputs.logits_per_text
    idx_max = np.argmax(logits_per_prompt.detach().cpu().numpy())
    return generated_labels[idx_max]


def cd_prompt(image, prompt, samples):
    cd_prompt =  (f"Imagine a photo with the caption '{prompt}' \
                    Now we want to create a new photo with a similar caption, but changing the \
                    appearance of the objects. Generate this caption.")

    disable_torch_init()
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {cd_prompt}"
    conv.append_message(conv.roles[0], inp)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=None,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    # output = outputs.rsplit('</s>', 1)[0]
    return output 

if __name__  == '__main__':
    task_config = 'augmentation-pipeline/configs/Random_Outpainting.yaml'

    task_config = load_yaml(task_config)
    measure_config = task_config['measurement']
    data_config = task_config["data"]

    images = get_dataset(data_config['image_path'])

    for image in images:
        prompt = "person" 

        # Load the image
        img,label_path = get_sample(image, data_config)
        
        # Get description of objects tracking
        objs = parse_xml_objects(label_path)

        prompt = get_prompt(img, objs, 10)
        print(prompt)

