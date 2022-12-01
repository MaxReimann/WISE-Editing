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
import json
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

import helpers.session_state as session_state 
from parameter_optimization.strotss_org import strotss, pil_resize_long_edge_to
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings, MinimalPipelineEffect 

st.set_page_config(layout="wide")
BASE_URL = "https://ivpg.hpi3d.de/wise/wise-demo/images/"
LOGGER = get_logger(__name__)


def upload_form(imgtype):
    with st.form(imgtype + "-form", clear_on_submit=True):
        uploaded_im = st.file_uploader(f"Load {imgtype} image:", type=["png", "jpg"], )
        upload_pressed = st.form_submit_button("Upload")

        if upload_pressed and uploaded_im is not None:
            img = Image.open(uploaded_im)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded = base64.b64encode(buffered.getvalue()).decode()
            # session_state.get(uploaded_im=img, content_render_src=f"data:image/jpeg;base64,{encoded}")
            session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_render_src": f"data:image/jpeg;base64,{encoded}",
                                    f"{imgtype}_id": "uploaded"})
            st.session_state["action"] = "uploaded"
            st.write("uploaded.")

upload_form("Content")
upload_form("Style")
content = st.session_state["Content_im"]
style = st.session_state["Style_im"]
base_url = "http://mr2632.byod.hpi.de:5000"

if content is not None and style is not None:
    optimize_button = st.sidebar.button("Optimize Style Transfer")
    if optimize_button:
        url = base_url + "/upload"
        content_path=f"/tmp/content-wise-uploaded{str(datetime.datetime.timestamp(datetime.datetime.now()))}.jpg"
        style_path=f"/tmp/content-wise-uploaded{str(datetime.datetime.timestamp(datetime.datetime.now()))}.jpg"
        content = pil_resize_long_edge_to(content, 1024)
        content.save(content_path)
        style = pil_resize_long_edge_to(style, 1024)
        style.save(style_path)
        files = {'style-image': open(style_path, "rb"), "content-image": open(content_path, "rb")}
        print("start-optimizing")
        task_id_res = requests.post(url, files=files)
        if task_id_res.status_code != 200:
            st.error(task_id_res.content)
            st.stop()
        else:
            task_id = task_id_res.json()['task_id']
        
        progress_bar = st.empty()
        with st.spinner(text="Optimizing parameters.."):
            started_time = time.time()
            while True:
                time.sleep(3)
                status = requests.get(base_url+"/get_status", params={"task_id": task_id})
                if status.status_code != 200:
                    print("get_status got status_code", status.status_code)
                    st.warning(status.content)
                    continue
                status = status.json()
                print(status)
                if status["status"] != "running" and status["status"] != "queued" :
                    if status["msg"] != "":
                        st.error(status["msg"])
                    break
                elif status["status"] == "queued":
                    started_time = time.time()
                    queue_length = requests.get(base_url+"/queue_length").json()
                    progress_bar.write(f"There are {queue_length['length']} tasks in the queue")
                elif status["progress"] == 0.0:
                    progressed = min(0.5 * (time.time() - started_time) / 80.0, 0.5) #estimate 80s for strotts
                    progress_bar.progress(progressed)
                else:
                    progress_bar.progress(min(0.5 + status["progress"] / 2.0, 1.0))
            vp_res = requests.get(base_url+"/get_vp", params={"task_id": task_id})
            if vp_res.status_code != 200:
                st.warning("got status" + str(vp_res.status_code))
                vp_res.raise_for_status()
            else:
                vp = np.load(BytesIO(vp_res.content))["vp"] 
                print("received vp from server")
                print("got numpy array", vp.shape)
                vp = torch.from_numpy(vp).cuda()

                effect, preset, param_set = get_default_settings("minimal_pipeline")
                effect.enable_checkpoints()
                effect.cuda()
                content_cuda = np_to_torch(content).cuda()
                with torch.no_grad():
                    result_cuda = effect(content_cuda, vp)
                img_res = Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8))
                st.image(img_res)
