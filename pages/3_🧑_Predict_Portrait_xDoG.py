import argparse
import base64
from io import BytesIO
from pathlib import Path
import os
import shutil
import sys
import time

import numpy as np
import torch.nn.functional as F
import torch
import streamlit as st
from st_click_detector import click_detector

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize
from PIL import Image

from huggingface_hub import hf_hub_download


PACKAGE_PARENT = '..'
WISE_DIR = '../wise/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, WISE_DIR)))


from local_ppn.options.test_options import TestOptions
from local_ppn.models import create_model

 
print(st.session_state["user"], " opened xDoG edits")

class CustomOpts(TestOptions):

    def remove_options(self, parser, options):
        for option in options:
            for action in parser._actions:
                print(action)
                if vars(action)['option_strings'][0] == option:
                    parser._handle_conflict_resolve(None,[(option,action)])
                    break
    
    def initialize(self, parser):
        parser = super(CustomOpts, self).initialize(parser)
        self.remove_options(parser, ["--dataroot"])
        return parser

    def print_options(self, opt):
        pass

def add_predefined_images():
    images = []
    for f in os.listdir(os.path.join(SCRIPT_DIR, PACKAGE_PARENT, 'images','apdrawing')):
        if not f.endswith('.jpg'):
            continue
        AB = Image.open(os.path.join(SCRIPT_DIR, PACKAGE_PARENT, 'images','apdrawing', f)).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        images.append(A)
    return images

@st.experimental_singleton
def make_model(_unused=None):
    model_path = hf_hub_download(repo_id="MaxReimann/WISE-APDrawing-XDoG", filename="apdrawing_xdog_ppn_conv.pth")
    os.makedirs(os.path.join(SCRIPT_DIR, PACKAGE_PARENT, "trained_models", "ours_apdrawing"), exist_ok=True)
    shutil.copy2(model_path, os.path.join(SCRIPT_DIR, PACKAGE_PARENT, "trained_models", "ours_apdrawing", "latest_net_G.pth"))

    opt = CustomOpts().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot ="null"
    opt.direction = "BtoA"
    opt.model = "pix2pix"
    opt.ppnG = "our_xdog"
    opt.name = "ours_apdrawing"
    opt.netG = "resnet_9blocks"
    opt.no_dropout = True 
    opt.norm = "batch"
    opt.load_size = 576
    opt.crop_size = 512
    opt.eval = False
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()


    return model, opt

def predict(image):
    model, opt = make_model()
    t = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    inp = image.resize((opt.crop_size, opt.crop_size), resample=Image.BICUBIC)
    inp = t(inp).unsqueeze(0).cuda()
    x = model.netG.module.ppn_part_forward(inp)

    output = model.netG.module.conv_part_forward(x)
    out_img = ToPILImage()(output.squeeze(0))
    return out_img



st.title("xDoG+CNN Portrait Drawing ")

images = add_predefined_images()

html_code = '<div class="column" style="display: flex; flex-wrap: wrap; padding: 0 4px;">'
for i, image in enumerate(images):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    html_code += f"<a href='#' id='{i}' style='padding: 0px 5px'><img height='120px' style='margin-top: 8px;' src='data:image/jpeg;base64,{encoded}'></a>"
html_code += "</div>"
clicked = click_detector(html_code)

uploaded_im = st.file_uploader(f"OR: Load portrait:", type=["png", "jpg"], )
if uploaded_im is not None:
    img = Image.open(uploaded_im)
    img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")


clicked_img = None
if clicked:
    clicked_img = images[int(clicked)]

sel_img = img if uploaded_im is not None else clicked_img
if sel_img:
    result_container = st.container()
    coll1, coll2 = result_container.columns([3,2])
    coll1.header("Result")
    coll2.header("Global Edits")
    
    model, opt = make_model()
    t = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    inp = sel_img.resize((opt.crop_size, opt.crop_size), resample=Image.BICUBIC)
    inp = t(inp).unsqueeze(0).cuda()
    # vp = model.netG.module.ppn_part_forward(inp)

    vp = model.netG.module.predict_parameters(inp)
    inp = (inp * 0.5) + 0.5

    effect = model.netG.module.apply_visual_effect.effect

    with coll2:
        # ("blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness")
        show_params_names = ["strokeWidth", "blackness", "contours"]
        display_means = []
        params_mapping = {"strokeWidth": ['strokeWidth'], 'blackness': ["blackness"], "contours": [ "details", "contour"]}
        def create_slider(name):
            params = params_mapping[name] if name in params_mapping else [name]
            means = [torch.mean(vp[:, effect.vpd.name2idx[n]]).item() for n in params]
            display_mean = float(np.average(means) + 0.5)
            display_means.append(display_mean)
            slider = st.slider(f"Mean {name}: ", 0.0, 1.0, value=display_mean, step=0.05)
            for i, param_name in enumerate(params):
                vp[:, effect.vpd.name2idx[param_name]] += slider - (means[i]+ 0.5)
                # vp.clamp_(-0.5, 0.5)
                # pass
        
        for name in show_params_names:
            create_slider(name)

        x = model.netG.module.apply_visual_effect(inp, vp)
        x = (x - 0.5) / 0.5
        
        only_x_dog = st.checkbox('only xdog', value=False, help='if checked, use only ppn+xdog, else use ppn+xdog+post-processing cnn')
        if only_x_dog:
            output = x[:,0].repeat(1,3,1,1)
            print('shape output', output.shape)
        else:
            output = model.netG.module.conv_part_forward(x)
    
    out_img = ToPILImage()(output.squeeze(0))
    output = out_img.resize((320,320), resample=Image.BICUBIC)
    with coll1:
        st.image(output)
