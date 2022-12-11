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
import streamlit as st
from demo_config import HUGGING_FACE, WORKER_URL



PACKAGE_PARENT = 'wise'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from parameter_optimization.parametric_styletransfer import single_optimize
from parameter_optimization.parametric_styletransfer import CONFIG as ST_CONFIG
from parameter_optimization.strotss_org import strotss, pil_resize_long_edge_to
from helpers import torch_to_np, np_to_torch

def retrieve_for_results_from_server():
    task_id = st.session_state['current_server_task_id']
    vp_res = requests.get(WORKER_URL+"/get_vp", params={"task_id": task_id})
    image_res = requests.get(WORKER_URL+"/get_image", params={"task_id": task_id})
    if vp_res.status_code != 200 or image_res.status_code != 200:
        st.warning("got status for " + WORKER_URL+"/get_vp" + str(vp_res.status_code))
        st.warning("got status for " + WORKER_URL+"/image_res" + str(image_res.status_code))
        st.session_state['current_server_task_id'] = None
        vp_res.raise_for_status()
        image_res.raise_for_status()
    else:
        st.session_state['current_server_task_id'] = None
        vp = np.load(BytesIO(vp_res.content))["vp"] 
        print("received vp from server")
        print("got numpy array", vp.shape)
        vp = torch.from_numpy(vp).cuda()
        image = Image.open(BytesIO(image_res.content))
        print("received image from server")
        image = np_to_torch(np.asarray(image)).cuda()

        st.session_state["effect_input"] = image
        st.session_state["result_vp"] = vp


def monitor_task(progress_placeholder):
    task_id = st.session_state['current_server_task_id']

    started_time = time.time()
    retries = 3
    with progress_placeholder.container():
        st.warning("Do not interact with the app until results are shown - otherwise results might be lost.")
        progress_bar = st.empty()
        while True:
            status = requests.get(WORKER_URL+"/get_status", params={"task_id": task_id})
            if status.status_code != 200:
                print("get_status got status_code", status.status_code)
                st.warning(status.content)
                retries -= 1
                if retries == 0:
                    return
                else:
                    time.sleep(2)
                    continue
            status = status.json()
            print(status)
            if status["status"] != "running" and status["status"] != "queued" :
                if status["msg"] != "":
                    print("got error for task", task_id, ":", status["msg"])
                    progress_placeholder.error(status["msg"])
                    st.session_state['current_server_task_id'] = None
                    st.stop()
                if status["status"] == "finished":
                    retrieve_for_results_from_server()
                return
            elif status["status"] == "queued":
                started_time = time.time()
                queue_length = requests.get(WORKER_URL+"/queue_length").json()
                progress_bar.write(f"There are {queue_length['length']} tasks in the queue")
            elif status["progress"] == 0.0:
                progressed = min(0.5 * (time.time() - started_time) / 80.0, 0.5) #estimate 80s for strotts
                progress_bar.progress(progressed)
            else:
                progress_bar.progress(min(0.5 + status["progress"] / 2.0, 1.0))

            time.sleep(2)

def get_queue_length():
    queue_length = requests.get(WORKER_URL+"/queue_length").json()
    return queue_length['length']


def optimize_on_server(content, style, result_image_placeholder):
    content_path=f"/tmp/content-wise-uploaded{str(datetime.datetime.timestamp(datetime.datetime.now()))}.jpg"
    style_path=f"/tmp/content-wise-uploaded{str(datetime.datetime.timestamp(datetime.datetime.now()))}.jpg"
    asp_c, asp_s =  content.height / content.width, style.height / style.width
    if any(a < 0.5 or a > 2.0 for a in (asp_c, asp_s)):
        result_image_placeholder.error('aspect ratio must be <= 2')
        st.stop()

    content = pil_resize_long_edge_to(content, 1024)
    content.save(content_path)
    style = pil_resize_long_edge_to(style, 1024)
    style.save(style_path)
    files = {'style-image': open(style_path, "rb"), "content-image": open(content_path, "rb")}
    print("start-optimizing. Time: ", datetime.datetime.now())
    url = WORKER_URL + "/upload"
    task_id_res = requests.post(url, files=files)
    if task_id_res.status_code != 200:
        result_image_placeholder.error(task_id_res.content)
        st.stop()
    else:
        task_id = task_id_res.json()['task_id']
        st.session_state['current_server_task_id'] = task_id

    monitor_task(result_image_placeholder)

def optimize_params(effect, preset, content, style, result_image_placeholder):
    result_image_placeholder.text("Executing NST to create reference image..")
    base_dir = f"result/{datetime.datetime.now().strftime(r'%Y-%m-%d %H.%Mh %Ss')}"
    os.makedirs(base_dir)
    reference = strotss(pil_resize_long_edge_to(content, 1024),
                        pil_resize_long_edge_to(style, 1024), content_weight=16.0,
                        device=torch.device("cuda"), space="uniform")
    progress_bar = result_image_placeholder.progress(0.0)
    ref_save_path = os.path.join(base_dir, "reference.jpg")
    content_save_path = os.path.join(base_dir, "content.jpg")
    resize_to = 720
    reference = pil_resize_long_edge_to(reference, resize_to)
    reference.save(ref_save_path)
    content.save(content_save_path)
    ST_CONFIG["n_iterations"] = 300
    
    vp, content_img_cuda = single_optimize(effect, preset, "l1", content_save_path, str(ref_save_path),
                                        write_video=False, base_dir=base_dir,
                                        iter_callback=lambda i: progress_bar.progress(
                                            float(i) / ST_CONFIG["n_iterations"]))
    st.session_state["effect_input"], st.session_state["result_vp"]  = content_img_cuda.detach(), vp.cuda().detach()
