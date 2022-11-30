import os
import sys
import torch.nn.functional as F
import torch

PACKAGE_PARENT = '../wise/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from effects.minimal_pipeline import MinimalPipelineEffect
from helpers.visual_parameter_def import minimal_pipeline_presets, minimal_pipeline_bump_mapping_preset, minimal_pipeline_xdog_preset
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings

st.set_page_config(page_title="Preset Edit Demo", layout="wide")


# @st.cache(hash_funcs={OilPaintEffect: id})
@st.cache(hash_funcs={MinimalPipelineEffect: id})
def local_edits_create_effect():
    effect, preset, param_set = get_default_settings("minimal_pipeline")
    effect.enable_checkpoints()
    effect.cuda()
    return effect, param_set


effect, param_set = local_edits_create_effect()
presets = {
    "original": minimal_pipeline_presets,
    "bump mapped": minimal_pipeline_bump_mapping_preset,
    "contoured": minimal_pipeline_xdog_preset
}

st.session_state["action"] = "switch_page_from_presets" # on switchback, remember effect input

active_preset = st.sidebar.selectbox("apply preset: ", ["original", "bump mapped", "contoured"])
blend_strength = st.sidebar.slider("Parameter blending strength (non-hue) : ", 0.0, 1.0, 1.0, 0.05)
hue_blend_strength = st.sidebar.slider("Hue-shift blending strength : ", 0.0, 1.0, 1.0, 0.05)

st.sidebar.text("Drawing options:")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 80, 40)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

st.session_state["preset_canvas_key"] ="preset_canvas"

vp = torch.clone(st.session_state["result_vp"])
org_cuda = st.session_state["effect_input"]

@st.experimental_memo
def greyscale_original(_org_cuda, content_id): #content_id is used for hashing
    if HUGGING_FACE:
        wsize = 450
        img_org_height, img_org_width = _org_cuda.shape[-2:]
        wpercent = (wsize / float(img_org_width))
        hsize = int((float(img_org_height) * float(wpercent)))
    else:
        longest_edge = 670
        img_org_height, img_org_width = _org_cuda.shape[-2:]
        max_width_height = max(img_org_width, img_org_height)
        hsize = int((float(longest_edge) * float(float(img_org_height) / max_width_height)))
        wsize = int((float(longest_edge) * float(float(img_org_width) / max_width_height)))

    org_img = F.interpolate(_org_cuda, (hsize, wsize), mode="bilinear")
    org_img = torch.mean(org_img, dim=1, keepdim=True) / 2.0
    org_img = torch_to_np(org_img, multiply_by_255=True)[..., np.newaxis].repeat(3, axis=2)
    org_img = Image.fromarray(org_img.astype(np.uint8))
    return org_img, hsize, wsize

greyscale_img, hsize, wsize = greyscale_original(org_cuda, st.session_state["Content_id"])

coll1, coll2 = st.columns(2)
coll1.header("Draw Mask")
coll2.header("Live Result")

with coll1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_image=greyscale_img,
        width=greyscale_img.width,
        height=greyscale_img.height,
        drawing_mode=drawing_mode,
        key=st.session_state["preset_canvas_key"]
    )
    

res_data = None
if canvas_result.image_data is not None:
    abc = np_to_torch(canvas_result.image_data.astype(np.float)).sum(dim=1, keepdim=True).cuda()

    img_org_width = org_cuda.shape[-1]
    img_org_height = org_cuda.shape[-2]
    res_data = F.interpolate(abc, (img_org_height, img_org_width)).squeeze(1)

    preset_tensor = effect.vpd.preset_tensor(presets[active_preset], org_cuda, add_local_dims=True)
    hue = torch.clone(vp[:,effect.vpd.name2idx["hueShift"]])
    vp[:] = preset_tensor * res_data * blend_strength + vp[:] * (1 - res_data * blend_strength)
    vp[:, effect.vpd.name2idx["hueShift"]] = \
        preset_tensor[:,effect.vpd.name2idx["hueShift"]] * res_data * hue_blend_strength + hue * (1 - res_data * hue_blend_strength)

with torch.no_grad():
    result_cuda = effect(org_cuda, vp)

img_res = Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8))
coll2.image(img_res)

apply_btn = st.sidebar.button("Apply")
if apply_btn:
    st.session_state["result_vp"] = vp