import numpy as np
import threading
from queue import Queue
from pycoral.adapters import common, classify
from pycoral.utils.edgetpu import make_interpreter

import utils


class Resnet50Tflite:
    def __init__(self, model_path):
        self.model_name = "resnet50"
        self.precision = "int8"
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        # keep input/output name to index mapping
        self.input2index = {i["name"]: i["index"] for i in self.interpreter.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.interpreter.get_output_details()}
        params = common.input_details(self.interpreter, 'quantization_parameters')
        self.input_scale = params['scales']
        self.input_zero_point = params['zero_points']
    
    def __call__(self, inputs):
        inputs = inputs / self.input_scale + self.input_zero_point
        inputs = inputs.astype(np.uint8)
        common.set_input(self.interpreter, inputs)
        self.interpreter.invoke()
        classes = classify.get_classes(self.interpreter, 1, 0.0)
        return classes
    
    def warmup(self, inputs, warmup_steps=100):
        for step in range(warmup_steps):
            self(inputs)
    
    def capture_stats(self):
        self.stop_event = threading.Event()
        self.output_queue = Queue()
        self.psutil_thread = threading.Thread(
            target=utils.get_psutil_stats, args=(self.output_queue, self.stop_event), daemon=True)
        self.psutil_thread.start()
    
    def get_avg_stats(self):
        ram_usage, cpu_util, gpu_util, temp = [], [], [], []
        
        while not self.output_queue.empty():
            c,r = self.output_queue.get()
            ram_usage.append(r)
            cpu_util.append(c)
        ram_usage, cpu_util = np.array(ram_usage), np.array(cpu_util)
        avg_ram, avg_cpu = np.sum(ram_usage) / len(ram_usage), \
            np.sum(cpu_util) / len(cpu_util)
        return round(avg_ram, 3), round(avg_cpu, 3), None, None
    
    def get_pred(self, outputs):
        return outputs[0].id
    
    def destroy(self):
        del self.interpreter

