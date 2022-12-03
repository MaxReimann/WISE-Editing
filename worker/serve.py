import datetime
import os
from pathlib import Path
import sys
from flask import Flask, jsonify, request, send_file, abort
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException, NotFound 
import json
import torch
import time
import threading
import traceback
from PIL import Image
import numpy as np

PACKAGE_PARENT = '..'
WISE_DIR = '../wise/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, WISE_DIR)))



from parameter_optimization.parametric_styletransfer import single_optimize
from parameter_optimization.parametric_styletransfer import CONFIG as ST_CONFIG
from parameter_optimization.strotss_org import strotss, pil_resize_long_edge_to
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings, MinimalPipelineEffect 

app = Flask(__name__)

image_folder = 'img_received'
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = image_folder
configure_uploads(app, photos)

class Args(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])
    def set_attributes(self, val_dict):
        for key in val_dict:
            setattr(self, key, val_dict[key])

default_args = {
        "output_image" : "output.jpg",
        ## values always set by request ##
        "content_image": "",
        "style_image": "",
        "output_vp": "",
        "iters": 500
}


total_task_count = 0

class NeuralOptimizer():
    def __init__(self, args) -> None:
        self.cur_iteration = 0
        self.args = args

    def optimize(self):
        base_dir = f"result/{datetime.datetime.now().strftime(r'%Y-%m-%d %H.%Mh %Ss')}"
        os.makedirs(base_dir)

        content = Image.open(self.args.content_image)
        style = Image.open(self.args.style_image)

        def set_iter(iter):
            self.cur_iteration=iter

        effect, preset, _ = get_default_settings("minimal_pipeline")
        effect.enable_checkpoints()

        reference = strotss(pil_resize_long_edge_to(content, 1024),
                            pil_resize_long_edge_to(style, 1024), content_weight=16.0,
                            device=torch.device("cuda"), space="uniform")

        ref_save_path = os.path.join(base_dir, "reference.jpg")
        resize_to = 720
        reference = pil_resize_long_edge_to(reference, resize_to)
        reference.save(ref_save_path)
        ST_CONFIG["n_iterations"] = self.args.iters
        vp, content_img_cuda = single_optimize(effect, preset, "l1", self.args.content_image, str(ref_save_path),
                                            write_video=False, base_dir=base_dir,
                                            iter_callback=set_iter)

        output = Image.fromarray(torch_to_np(content_img_cuda.detach().cpu() * 255.0).astype(np.uint8))
        output.save(self.args.output_image)
        # torch.save (vp.detach().clone(), self.args.output_vp)
        # preset_tensor = effect.vpd.preset_tensor(preset, np_to_torch(np.array(content)).cuda(), add_local_dims=True)
        np.savez_compressed(self.args.output_vp, vp=vp.detach().cpu().numpy())

        

class StyleTask:
    def __init__(self, task_id, style_filename, content_filename):
        self.content_filename=content_filename
        self.style_filename=style_filename

        self.status = "queued"
        self.task_id = task_id
        self.error_msg = ""
        self.output_filename = content_filename.split(".")[0] + "_output.jpg"
        self.vp_output_filename = content_filename.split(".")[0] + "_output.npz"

        # global neural_optimizer
        # if neural_optimizer is None:
        #     neural_optimizer = NeuralOptimizer(Args(default_args))

        self.neural_optimizer = NeuralOptimizer(Args(default_args))
    
    def start(self):
        self.neural_optimizer.args.set_attributes(default_args)

        self.neural_optimizer.args.style_image = os.path.join(image_folder, self.style_filename)
        self.neural_optimizer.args.content_image = os.path.join(image_folder, self.content_filename)
        self.neural_optimizer.args.output_image = os.path.join(image_folder, self.output_filename)
        self.neural_optimizer.args.output_vp = os.path.join(image_folder, self.vp_output_filename)
 
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self):
        self.status = "running"
        try:
            self.neural_optimizer.optimize()
        except Exception as e:
            print("Error in task %d :"%(self.task_id), str(e))
            traceback.print_exc()

            self.status = "error"
            self.error_msg = str(e)
            return

        self.status = "finished"
        del self.neural_optimizer
        torch.cuda.empty_cache() 
        print("finished styling task: " + str(self.task_id))

class StylerQueue:
    queued_tasks = []
    finished_tasks = []
    running_task = None

    def __init__(self):
        thread = threading.Thread(target=self.status_checker, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def queue_task(self, *args):
        global total_task_count
        total_task_count += 1
        task_id = abs(hash(str(time.time())))
        print("queued task num. ", total_task_count, "with ID", task_id)
        task = StyleTask(task_id, *args)
        self.queued_tasks.append(task)

        return task_id

    def get_task(self, task_id):
        if self.running_task is not None and self.running_task.task_id == task_id:
            return self.running_task
        task = next((task for task in self.queued_tasks + self.finished_tasks if task.task_id == task_id), None)
        return task

    def status_checker(self):
        while True:
            time.sleep(0.3)

            if self.running_task is None:
                if len(self.queued_tasks) > 0:
                    self.running_task = self.queued_tasks[0]
                    self.running_task.start()
                    self.queued_tasks = self.queued_tasks[1:]
            elif self.running_task.status == "finished" or self.running_task.status == "error": 
                self.finished_tasks.append(self.running_task)
                if len(self.queued_tasks) > 0:
                    self.running_task = self.queued_tasks[0]
                    self.running_task.start()
                    self.queued_tasks = self.queued_tasks[1:]
                else:
                    self.running_task = None

styler_queue = StylerQueue()

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(message=str(e)), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(message=str(e)), 500

@app.errorhandler(400)
def caught_error(e, *args):
    print(args)
    print(e)
    return jsonify(message=str(e.description)), 400

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    abort(404, "route not found")



@app.route('/upload', methods=['POST'])
def upload():
    if 'style-image' in request.files and \
        'content-image' in request.files:

        style_filename = photos.save(request.files['style-image'])
        content_filename = photos.save(request.files['content-image'])

        job_id = styler_queue.queue_task(style_filename, content_filename)
        print('added new stylization task', style_filename, content_filename)

        return jsonify({"task_id": job_id})
    abort(400, description="request needs style, content image")

@app.route('/get_status')
def get_status():
    if request.args.get("task_id") is None:
        abort(400, description="task_id needs to be supplied as parameter")
    task_id = int(request.args.get("task_id"))
    task = styler_queue.get_task(task_id)

    if task is None:
        abort(400, description="task with id %d not found"%task_id)

    status = {
        "status": task.status,
        "msg": task.error_msg
    }

    if task.status == "running":
        if isinstance(task, StyleTask):
            status["progress"] = float(task.neural_optimizer.cur_iteration) / float(default_args["iters"]) 

    return jsonify(status)

@app.route('/queue_length')
def get_queue_length():
    tasks = len(styler_queue.queued_tasks)
    if styler_queue.running_task is not None:
        tasks += 1

    status = {
        "length": tasks
    }

    return jsonify(status)


@app.route('/get_image')
def get_image():
    if request.args.get("task_id") is None:
        abort(400, description="task_id needs to be supplied as parameter")
    task_id = int(request.args.get("task_id"))
    task = styler_queue.get_task(task_id)

    if task is None:
        abort(400, description="task with id %d not found"%task_id)

    if task.status != "finished":
        abort(400, description="task with id %d not in finished state"%task_id)

    return send_file(os.path.join(image_folder, task.output_filename), mimetype='image/jpg')

@app.route('/get_vp')
def get_vp():
    if request.args.get("task_id") is None:
        abort(400, description="task_id needs to be supplied as parameter")
    task_id = int(request.args.get("task_id"))
    task = styler_queue.get_task(task_id)

    if task is None:
        abort(400, description="task with id %d not found"%task_id)

    if task.status != "finished":
        abort(400, description="task with id %d not in finished state"%task_id)

    return send_file(os.path.join(image_folder, task.vp_output_filename), mimetype='application/zip')


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0",port=8600)
