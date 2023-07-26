import argparse
import time
import psutil
import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa # pylint: disable=unused-import
import pycuda.driver as cuda
import threading

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

cpu_usage = []
ram_consumption = []
stop_usage_capture = False


def preprocess_img(filename):
    input_image = Image.open(filename)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.numpy()


class Resnet50TRT:
    """Resnet50 TensorRT inference utility class.
    """
    def __init__(self, engine_path):
        """Initialize.
        """
        # Create a Context on this device,
        self._ctx = cuda.Device(0).make_context()
        self._logger = trt.Logger(trt.Logger.INFO)
        self._stream = cuda.Stream()

        # initiate engine related class attributes
        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None

        self._load_model(engine_path)
        self._allocate_buffers()

    def _deserialize_engine(self, trt_engine_path: str) -> trt.tensorrt.ICudaEngine:
        """Deserialize TensorRT Cuda Engine
        Args:
            trt_engine_path (str): path to engine file
        Returns:
            trt.tensorrt.ICudaEngine: deserialized engine
        """
        with open(trt_engine_path, 'rb') as engine_file:
            with trt.Runtime(self._logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_file.read())

        return engine
    
    def _allocate_buffers(self) -> None:
        """Allocates memory for inference using TensorRT engine.
        """
        inputs, outputs, bindings = [], [], []
        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        # set buffers
        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings

    def _load_model(self, engine_path):
        print("[INFO] Deserializing TensorRT engine ...")
        # build engine with given configs and load it
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine does not exist {engine_path}.")

        # deserialize and load engine
        self._engine = self._deserialize_engine(engine_path)

        if not self._engine:
            raise Exception("[Error] Couldn't deserialize engine successfully !")

        # create execution context
        self._context = self._engine.create_execution_context()
        if not self._context:
            raise Exception(
                "[Error] Couldn't create execution context from engine successfully !")

    def __call__(self, inputs: np.ndarray):
        """Runs inference on the given inputs.
        Args:
            inputs (np.ndarray): channels-first format,
            with/without batch axis
        Returns:
            List[np.ndarray]: inference's output (raw tensorrt output)

        """
        self._ctx.push()

        # copy inputs to input memory
        # without astype gives invalid arg error
        self._inputs[0]['host'] = inputs.ravel()

        # transfer data to the gpu
        cuda.memcpy_htod_async(
            self._inputs[0]['device'], self._inputs[0]['host'], self._stream)
        
        # run inference
        self._context.execute_async(
            batch_size=1,
            bindings=self._bindings,
            stream_handle=self._stream.handle)

        # fetch outputs from gpu
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)

        # synchronize stream
        self._stream.synchronize()
        self._ctx.pop()
        return [out['host'] for out in self._outputs]

    def destroy(self):
        """Destroy if any context in the stack.
        """
        try:
            self._ctx.pop()
        except Exception as exception:
            pass


def capture_cpu_usage():
    global cpu_usage, stop_usage_capture
    while not stop_usage_capture:
        cpu_usage.append(psutil.cpu_percent(interval=None))
        time.sleep(0.5)


def capture_ram():
    global ram_consumption, stop_usage_capture
    while not stop_usage_capture:
        # Getting RAM usage in MB
        ram_consumption.append(psutil.virtual_memory()[3] / 1000000)
        time.sleep(0.5)

def main(args):
    global stop_usage_capture, cpu_usage, ram_consumption
    print(args)
    os.makedirs(args.preprocessed_dir, exist_ok=True)

    # preprocess and save np array
    jpeg_files_list = os.listdir(args.imagenet)
    
    for filename in tqdm(jpeg_files_list, desc="Preprocessing", unit="image"):
        if not filename.lower().endswith('.jpeg'):
            continue
        jpeg_path = os.path.join(args.imagenet, filename)
        npy_path = os.path.join(args.preprocessed_dir, filename.replace("JPEG", "npy"))
        if os.path.exists(npy_path):
            continue
        
        preprocessed = preprocess_img(jpeg_path)
        np.save(npy_path, preprocessed)
    
    # load val_map
    labels = {}
    with open(os.path.join(args.imagenet, 'val_map.txt'), "r") as f:
        lines = f.readlines()
        for line in lines:
            p, l = line.split(' ')
            labels[p.split('.')[0]] = int(l)
    times = []
    accuracy = []

    # inference
    r50_trt_engine = Resnet50TRT(args.engine_path)
    
    # warmup
    data = np.ones((1 * 3 * 224 * 224), dtype=np.float32)
    for i in range(100):
        outputs = r50_trt_engine(data)
    
    npy_arrays = os.listdir(args.preprocessed_dir)

    # start ram, cpu usage threads
    cpu_thread = threading.Thread(target=capture_cpu_usage)
    ram_thread = threading.Thread(target=capture_ram)

    cpu_thread.start()
    ram_thread.start()

    for npy_arr in tqdm(npy_arrays,  desc="Running inference"):
        inputs = np.load(os.path.join(args.preprocessed_dir, npy_arr))
        start = time.time()
        outputs = r50_trt_engine(inputs)
        t = time.time() - start   
        times.append(t)     
        pred = np.array(outputs).argmax()
        gt = labels[npy_arr.split('.')[0]]
        if pred==gt:
            accuracy.append(1)
        else:
            accuracy.append(0)
        
    r50_trt_engine.destroy()
    stop_usage_capture = True
    np_acc = np.array(accuracy)
    np_lat = np.array(times)
    print(f"Accuracy = {np.count_nonzero(np_acc == 1)/len(np_acc)}")
    print(f"Latency = {np.sum(np_lat)/len(np_lat)}")

    print(np.array(cpu_usage).sum()/len(cpu_usage))
    print(np.array(ram_consumption).sum()/len(ram_consumption))

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet",
        default='/mnt/val',
        type=str,
        help="directory with imagenet images"
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="/mnt/workspace/imagenet_preprocessed",
        type=str,
        help="directory to store preprocessed np array"
    )
    parser.add_argument(
        "--engine-path",
        default="/mnt/workspace/tensorrtx/resnet/build/resnet50_fp16.engine",
        type=str,
        help="tensorrt engine path"
    )
    args = parser.parse_args()
    main(args)
