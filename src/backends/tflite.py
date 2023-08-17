import numpy as np
import threading
from queue import Queue
from pycoral.adapters import common, classify
from pycoral.utils.edgetpu import make_interpreter

import utils
from backends.backend import Backend


class TfliteBackend(Backend):
    def __init__(self, name, device="tpu"):
        super(TfliteBackend, self).__init__(name)
        self.precision = "int8" if device=="tpu" else "fp32"
    
    def name(self):
        return self.name
    
    def version(self):
        import tflite_runtime
        return tflite_runtime.__version__

    def load_backend(self, model_path, model_name=None):
        self.model_name = model_name
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
            target=utils.get_coral_stats, args=(self.output_queue, self.stop_event), daemon=True
        )
        self.psutil_thread.start()
    
    def get_avg_stats(self):
        ram_usage, cpu_util, temp, tpu_freq, cpu_freq = [], [], [], [], []
        
        while not self.output_queue.empty():
            c,r,t, tf, cf = self.output_queue.get()
            ram_usage.append(r)
            cpu_util.append(c)
            temp.append(t)
            tpu_freq.append(tf)
            cpu_freq.append(cf)
        ram_usage, cpu_util, temp = np.array(ram_usage), np.array(cpu_util), np.array(temp)
        stats = {
            "cpu": cpu_util,
            "memory": ram_usage,
            "temperature": temp,
            "tpu_freq": tpu_freq,
            "cpu_freq": cpu_freq
        }
        return stats
    
    def get_pred(self, outputs):
        return outputs[0].id
    
    def destroy(self):
        del self.interpreter

