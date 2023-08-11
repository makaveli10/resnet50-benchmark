import argparse
import utils
from tqdm import tqdm
import os
import numpy as np
import time


def main(args):
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
        if args.backend == "tflite":
            preprocessed = utils.pre_process_tflite(jpeg_path)
        else:
            preprocessed = utils.preprocess_img(jpeg_path)
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

    backend = None
    data = None
    if args.backend == "tensorrt":
        from backends.tensorrt import TRTBackend
        precision = "fp16" if "fp16" in args.model_path else "fp32"
        backend = TRTBackend(name="tensorrt", precision=precision)
        data = np.ones((1 * 3 * 224 * 224), dtype=np.float32)
    
    if args.backend == "tflite":
        from backends.tflite import TfliteBackend
        backend = TfliteBackend(name="tflite")
        data = np.ones((224, 224, 3), dtype=np.float32)
    
    if args.backend == "ncnn":
        from backends.ncnn import NCNNBackend
        backend = NCNNBackend()
        data = np.ones((1, 3, 224, 224), dtype=np.float32)

    if backend is None:
        print("backend is none")
        return
    
    backend.load_backend(args.model_path, model_name=args.model_name)
    backend.warmup(data)

    backend.capture_stats()

    npy_arrays = os.listdir(args.preprocessed_dir)
    if args.count is not None:
        npy_arrays = npy_arrays[:args.count]

    for npy_arr in tqdm(npy_arrays,  desc="Running inference"):
        inputs = np.load(os.path.join(args.preprocessed_dir, npy_arr))
        start = time.time()
        outputs = backend(inputs)
        t = time.time() - start   
        times.append(t)
        pred = backend.get_pred(outputs)
        gt = labels[npy_arr.split('.')[0]]
        if pred==gt:
            accuracy.append(1)
        else:
            accuracy.append(0)
    
    backend.stop_event.set()
    backend.destroy()

    np_acc = np.array(accuracy)
    np_lat = np.array(times)
    print(f"Accuracy = {np.count_nonzero(np_acc == 1)/len(np_acc)}")
    print(f"Latency = {np.sum(np_lat)/len(np_lat)}") 
      
    ram_usage, cpu_util, gpu_util, temp = backend.get_avg_stats()

    data_dict = {}
    data_dict["system"] = utils.get_device_model()
    data_dict["processor"] = utils.get_cpu()
    data_dict["accelerator"] = utils.build_and_run_device_query() if args.backend in ["tensorrt"] else ""
    data_dict["model_name"] = backend.model_name
    data_dict["framework"] = f"{args.backend}"
    data_dict["latency"] = round(float(np.sum(np_lat)/len(np_lat))*1000, 3)
    data_dict["precision"] = backend.precision
    data_dict["accuracy"] = round(float(np.count_nonzero(np_acc == 1)/len(np_acc))*100, 3)
    data_dict["cpu"] = float(round(np.average(cpu_util), 2))
    data_dict["memory"] = float(round(np.average(ram_usage), 2))
    data_dict["power"] = ""
    data_dict["temperature"] = "" if temp is None else float(temp)
    print(data_dict)

    # TODO: send data dict to db

    # TODO: send cpu_util(has percentage usage per core for the whole run)

    # TODO: send ram_usage(has ram usage in MB for the whole run)
    

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet",
        default='/mnt/workspace/imagenet-2012/val',
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
        "--backend",
        default=None,
        type=str,
        help="backend name"
    )
    parser.add_argument(
        "--model-name",
        default="resnet50",
        type=str,
        help="model name"
    )
    parser.add_argument(
        "--model_path",
        default="/mnt/workspace/CM/repos/local/cache/dc84a916f80f41d1/resnet50_v1",
        type=str,
        help="model path"
    )
    parser.add_argument(
        "--count",
        default=None,
        type=int,
        help="no of images to run benchmark on"
    )
    args = parser.parse_args()
    main(args)