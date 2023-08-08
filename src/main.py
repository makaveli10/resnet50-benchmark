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

    model = None
    data = None
    if args.backend == "tensorrt":
        from backend.tensorrt import Resnet50TRT

        model = Resnet50TRT(args.model_path)
        data = np.ones((1 * 3 * 224 * 224), dtype=np.float32)
    
    if args.backend == "tflite":
        from backend.tflite import Resnet50Tflite
        print(args.model_path)
        model = Resnet50Tflite(args.model_path)
        data = np.ones((224, 224, 3), dtype=np.float32)

    if model is None:
        print("model is none")
        return

    # warmup
    model.warmup(data)

    model.capture_stats()

    npy_arrays = os.listdir(args.preprocessed_dir)

    for npy_arr in tqdm(npy_arrays,  desc="Running inference"):
        inputs = np.load(os.path.join(args.preprocessed_dir, npy_arr))
        start = time.time()
        outputs = model(inputs)
        t = time.time() - start   
        times.append(t)
        pred = model.get_pred(outputs)
        gt = labels[npy_arr.split('.')[0]]
        if pred==gt:
            accuracy.append(1)
        else:
            accuracy.append(0)
    
    model.stop_event.set()
    model.destroy()

    np_acc = np.array(accuracy)
    np_lat = np.array(times)
    print(f"Accuracy = {np.count_nonzero(np_acc == 1)/len(np_acc)}")
    print(f"Latency = {np.sum(np_lat)/len(np_lat)}") 
      
    ram_usage, cpu_util, gpu_util, temp = model.get_avg_stats()
    print(ram_usage, cpu_util, gpu_util, temp)

    data_dict = {}
    data_dict["system"] = utils.get_device_model()
    data_dict["processor"] = utils.get_cpu()
    data_dict["accelerator"] = utils.build_and_run_device_query() if args.backend in ["tensorrt"] else ""
    data_dict["model_name"] = model.model_name
    data_dict["framework"] = f"{args.backend}"
    data_dict["latency"] = round(float(np.sum(np_lat)/len(np_lat))*1000, 3)
    data_dict["precision"] = model.precision
    data_dict["accuracy"] = round(float(np.count_nonzero(np_acc == 1)/len(np_acc))*100, 3)
    data_dict["cpu"] = float(cpu_util)
    data_dict["memory"] = int(ram_usage)
    data_dict["power"] = ""
    data_dict["temperature"] = "" if temp is None else float(temp)
    print(data_dict)
        
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
        "--backend",
        default=None,
        type=str,
        help="backend name"
    )
    parser.add_argument(
        "--model_path",
        default="/mnt/workspace/tensorrtx/resnet/build/resnet50_fp16.engine",
        type=str,
        help="model path"
    )
    args = parser.parse_args()
    main(args)