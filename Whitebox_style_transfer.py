import base64
import datetime
import os
import sys
from io import BytesIO
from pathlib import Path
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
import time

PACKAGE_PARENT = 'wise'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import streamlit as st
from streamlit.logger import get_logger
from st_click_detector import click_detector
import streamlit.components.v1 as components
from streamlit.source_util import get_pages
from streamlit_extras.switch_page_button import switch_page

from demo_config import HUGGING_FACE
from parameter_optimization.parametric_styletransfer import single_optimize
from parameter_optimization.parametric_styletransfer import CONFIG as ST_CONFIG
from parameter_optimization.strotss_org import strotss, pil_resize_long_edge_to
import helpers.session_state as session_state
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings, MinimalPipelineEffect 

st.set_page_config(layout="wide")
BASE_URL = "https://ivpg.hpi3d.de/wise/wise-demo/images/"
LOGGER = get_logger(__name__)

effect_type = "minimal_pipeline"

if "click_counter" not in st.session_state:
    st.session_state.click_counter = 1

if "action" not in st.session_state:
    st.session_state["action"] = ""
    
if "user" not in st.session_state:
    st.session_state["user"] = hash(time.time())

content_urls = [
    {
        "name": "Portrait", "id": "portrait",
        "src": BASE_URL + "/content/portrait.jpeg"
    },
    {
        "name": "Tuebingen", "id": "tubingen",
        "src": BASE_URL + "/content/tubingen.jpeg"
    },
    {
        "name": "Colibri", "id": "colibri",
        "src": BASE_URL + "/content/colibri.jpeg"
    }
]

style_urls = [
    {
        "name": "Starry Night, Van Gogh", "id": "starry_night",
        "src": BASE_URL + "/style/starry_night.jpg"
    },
    {
        "name": "The Scream, Edward Munch", "id": "the_scream",
        "src": BASE_URL + "/style/the_scream.jpg"
    },
    {
        "name": "The Great Wave, Ukiyo-e", "id": "wave",
        "src": BASE_URL + "/style/wave.jpg"
    },
    {
        "name": "Woman with Hat, Henry Matisse", "id": "woman_with_hat",
        "src": BASE_URL + "/style/woman_with_hat.jpg"
    }
]


def last_image_clicked(type="content", action=None, ):
    kw = "last_image_clicked" + "_" + type
    if action:
        session_state.get(**{kw: action})
    elif kw not in session_state.get():
        return None
    else:
        return session_state.get()[kw]


@st.cache
def _retrieve_from_id(clicked, urls):
    src = [x["src"] for x in urls if x["id"] == clicked][0]
    img = Image.open(requests.get(src, stream=True).raw)
    return img, src


def store_img_from_id(clicked, urls, imgtype):
    img, src = _retrieve_from_id(clicked, urls)
    session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_render_src": src, f"{imgtype}_id": clicked})


def img_choice_panel(imgtype, urls, default_choice, expanded):
    with st.expander(f"Select {imgtype} image:", expanded=expanded):
        html_code = '<div class="column" style="display: flex; flex-wrap: wrap; padding: 0 4px;">'
        for url in urls:
            html_code += f"<a href='#' id='{url['id']}' style='padding: 0px 5px'><img height='160px' style='margin-top: 8px;' src='{url['src']}'></a>"
        html_code += "</div>"
        clicked = click_detector(html_code)

        if not clicked and st.session_state["action"] not in ("uploaded", "switch_page_from_local_edits", "switch_page_from_presets", "slider_change", "reset"):  # default val
            store_img_from_id(default_choice, urls, imgtype)

        st.write("OR:  ")

        with st.form(imgtype + "-form", clear_on_submit=True):
            uploaded_im = st.file_uploader(f"Load {imgtype} image:", type=["png", "jpg"], )
            upload_pressed = st.form_submit_button("Upload")

            if upload_pressed and uploaded_im is not None:
                img = Image.open(uploaded_im)
                img = img.convert('RGB')
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                encoded = base64.b64encode(buffered.getvalue()).decode()
                # session_state.get(uploaded_im=img, content_render_src=f"data:image/jpeg;base64,{encoded}")
                session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_render_src": f"data:image/jpeg;base64,{encoded}",
                                     f"{imgtype}_id": "uploaded"})
                st.session_state["action"] = "uploaded"
                st.write("uploaded.")

        last_clicked = last_image_clicked(type=imgtype)
        print(st.session_state["user"], " last_clicked", last_clicked, "clicked", clicked, "action", st.session_state["action"] )
        if not upload_pressed and clicked != "":  # trigger when no file uploaded
            if last_clicked != clicked:  # only activate when content was actually clicked
                store_img_from_id(clicked, urls, imgtype)
                last_image_clicked(type=imgtype, action=clicked)
                st.session_state["action"] = "clicked"
                st.session_state.click_counter += 1  # hack to get page to reload at top

        state = session_state.get()
        st.sidebar.write(f'Selected {imgtype} image:')
        st.sidebar.markdown(f'<img src="{state[f"{imgtype}_render_src"]}" width=240px></img>', unsafe_allow_html=True)

def optimize(effect, preset, result_image_placeholder):
    content = st.session_state["Content_im"]
    style = st.session_state["Style_im"]
    st.session_state["optimize_next"] = False
    with st.spinner(text="Optimizing parameters.."):
        print("optimizing for user", st.session_state["user"])
        if HUGGING_FACE:
            optimize_on_server(content, style, result_image_placeholder)
        else:
            optimize_params(effect, preset, content, style, result_image_placeholder)

def optimize_next(result_image_placeholder):
    result_image_placeholder.text("<- Custom content/style needs to be style transferred")
    queue_length = 0 if not HUGGING_FACE else get_queue_length()
    if queue_length > 0:
        st.sidebar.warning(f"WARNING: Already {queue_length} tasks in the queue. It will take approx {(queue_length+1) * 5} min for your image to be completed.")
    else:
        st.sidebar.warning("Note: Optimizing takes up to 5 minutes.")
    optimize_button = st.sidebar.button("Optimize Style Transfer")
    if optimize_button:
        st.session_state["optimize_next"] = True
        st.experimental_rerun()
    else:
        if not "result_vp" in st.session_state:
            st.stop()
        else:
            return st.session_state["effect_input"], st.session_state["result_vp"]


@st.cache(hash_funcs={MinimalPipelineEffect: id})
def create_effect():
    effect, preset, param_set = get_default_settings(effect_type)
    effect.enable_checkpoints()
    effect.cuda()
    return effect, preset


def load_visual_params(vp_path: str, img_org: Image, org_cuda: torch.Tensor, effect) -> torch.Tensor:
    if Path(vp_path).exists():
        vp = torch.load(vp_path).detach().clone()
        vp = F.interpolate(vp, (img_org.height, img_org.width))
        if len(effect.vpd.vp_ranges) == vp.shape[1]:
            return vp
    # use preset and save it
    vp = effect.vpd.preset_tensor(preset, org_cuda, add_local_dims=True)
    torch.save(vp, vp_path)
    return vp


# @st.cache(hash_funcs={torch.Tensor: id})
@st.experimental_memo
def load_params(content_id, style_id):#, effect):
    preoptim_param_path = os.path.join("precomputed", effect_type, content_id, style_id)
    img_org = Image.open(os.path.join(preoptim_param_path, "input.png"))
    content_cuda = np_to_torch(img_org).cuda()
    vp_path = os.path.join(preoptim_param_path, "vp.pt")
    vp = load_visual_params(vp_path, img_org, content_cuda, effect)
    return content_cuda, vp


def render_effect(effect, content_cuda, vp):
    with torch.no_grad():
        result_cuda = effect(content_cuda, vp)
    img_res = Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8))
    return img_res


result_container = st.container()
coll1, coll2 = result_container.columns([3,2])
coll1.header("Result")
coll2.header("Global Edits")
result_image_placeholder = coll1.empty()
result_image_placeholder.markdown("## loading..")

from tasks import optimize_on_server, optimize_params, monitor_task, get_queue_length

if "current_server_task_id" not in st.session_state:
    st.session_state['current_server_task_id'] = None

if "optimize_next" not in st.session_state:
    st.session_state['optimize_next'] = False

effect, preset = create_effect()

if HUGGING_FACE and st.session_state['current_server_task_id'] is not None: 
    with st.spinner(text="Optimizing parameters.."):
        monitor_task(result_image_placeholder)

if st.session_state["optimize_next"]:
    print("optimize now")
    optimize(effect, preset, result_image_placeholder)

img_choice_panel("Content", content_urls, "portrait", expanded=True)
img_choice_panel("Style", style_urls, "starry_night", expanded=True)

state = session_state.get()
content_id = state["Content_id"]
style_id = state["Style_id"]


print("content id, style id", content_id, style_id  )
if st.session_state["action"] == "uploaded":
    content_img, _vp = optimize_next(result_image_placeholder)
elif st.session_state["action"] in ("switch_page_from_local_edits", "switch_page_from_presets", "slider_change") or \
      content_id == "uploaded" or style_id == "uploaded":
    print(st.session_state["user"], "restore param")
    _vp = st.session_state["result_vp"]
    content_img = st.session_state["effect_input"]
else:
    print(st.session_state["user"], "load_params")
    content_img, _vp = load_params(content_id, style_id)#, effect)

vp = torch.clone(_vp)


def reset_params(means, names):
    for i, name in enumerate(names):
        st.session_state["slider_" + name] = means[i]

def on_slider():
    st.session_state["action"] = "slider_change"


with coll2:
    show_params_names = [ 'bumpiness',"bumpSpecular", "contours"]
    display_means = []
    params_mapping = {"bumpiness": ['bumpScale', "bumpOpacity"], "bumpSpecular": ["bumpSpecular"], "contours": [ "contourOpacity", "contour"]}
    def create_slider(name):
        params = params_mapping[name] if name in params_mapping else [name]
        means = [torch.mean(vp[:, effect.vpd.name2idx[n]]).item() for n in params]
        display_mean = np.average(means) + 0.5
        display_means.append(display_mean)
        if "slider_" + name not in st.session_state or st.session_state["action"] != "slider_change": 
          st.session_state["slider_" + name] = display_mean
        slider = st.slider(f"Mean {name}: ", 0.0, 1.0, step=0.05, key="slider_" + name, on_change=on_slider)
        for i, param_name in enumerate(params):
            vp[:, effect.vpd.name2idx[param_name]] += slider - (means[i] + 0.5)
            vp.clamp_(-0.5, 0.5)
    
    for name in show_params_names:
        create_slider(name)

    others_idx = set(range(len(effect.vpd.vp_ranges))) - set([effect.vpd.name2idx[name] for name in sum(params_mapping.values(), [])])
    others_names = [effect.vpd.vp_ranges[i][0] for i in sorted(list(others_idx))]
    other_param = st.selectbox("Other parameters: ", ["hueShift"] + [n for n in others_names if n != "hueShift"] )
    create_slider(other_param)


    reset_button = st.button("Reset Parameters", on_click=reset_params, args=(display_means, show_params_names))
    if reset_button:
        st.session_state["action"] = "reset"
        st.experimental_rerun()

    apply_presets = st.button("Paint Presets")
    if apply_presets:
        switch_page("Apply_preset")

    edit_locally_btn = st.button("Edit Local Parameter Maps")
    if edit_locally_btn:
        switch_page('️ local edits')



img_res = render_effect(effect, content_img, vp)

st.session_state["result_vp"] = vp
st.session_state["effect_input"] = content_img
st.session_state["last_result"] = img_res

with coll1:
    result_image_placeholder.image(img_res)

# a bit hacky way to return focus to top of page after clicking on images
components.html(
    f"""
        <p>{st.session_state.click_counter}</p>
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """,
    height=0
)
