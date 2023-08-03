import os
import sys
import subprocess
import re
import threading
from torchvision import transforms
from PIL import Image



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


def extract_tegrastats_info(output):
    # e.g. RAM 1542/1980MB
    ram_match = re.search(r'RAM (\d+)/\d+MB', output)
    if ram_match:
        ram_usage = int(ram_match.group(1))
    else:
        ram_usage = None

    # CPU utilization (e.g., "CPU [28%@614,16%@614,12%@614,19%@614]")
    cpu_util_match = re.search(r'CPU \[([\d%@,]+)\]', output)
    if cpu_util_match:
        cpu_values = [int(val.split('%')[0]) for val in cpu_util_match.group(1).split(',')]
        cpu_utilization = sum(cpu_values) / len(cpu_values)
    else:
        cpu_utilization = None

    # GPU (e.g. GR3D_FREQ 61%)
    gpu_util_match = re.search(r'GR3D_FREQ (\d+)%', output)
    if gpu_util_match:
        gpu_utilisation = int(gpu_util_match.group(1))
    else:
        gpu_utilisation = None

    # Temperature (e.g., "thermal@34.5C")
    temp_match = re.search(r'thermal@([\d.]+)C', output)
    if temp_match:
        temperature = float(temp_match.group(1))
    else:
        temperature = None

    return ram_usage, cpu_utilization, gpu_utilisation, temperature


def read_tegrastats_output(process, output_queue, stop_event):
    while not stop_event.is_set():
        output = process.stdout.readline()
        if not output:
            break
        ram_usage, cpu_utilization, gpu_utilization, temperature = extract_tegrastats_info(output.decode().strip())
        if ram_usage is not None or cpu_utilization is not None or \
         gpu_utilization is not None or temperature is not None:
            output_queue.put((ram_usage, cpu_utilization, gpu_utilization, temperature))
    process.terminate()


def parse_lscpu_output(output):
    data = {}
    lines = output.decode().splitlines() \
        if isinstance(output, bytes) else output.splitlines()
    for line in lines:
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def get_cpu():
    try:
        if sys.version_info.major == 3 and sys.version_info.minor == 6:
            # Run the lscpu command with subprocess.check_output() for Python 3.6
            result = subprocess.check_output(['lscpu'])
        else:
            # Run the lscpu command with subprocess.run() for Python 3.7 and above
            result = subprocess.run(['lscpu'], stdout=subprocess.PIPE).stdout
        parsed_out = parse_lscpu_output(result)
        return parsed_out['Model name']
    except Exception as e:
        print(f"Error: {e}")
    

def get_device_model():
    try:
        result = subprocess.run(['cat', '/proc/device-tree/model'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        output = result.stdout.decode().strip('\x00')
        return output
    except FileNotFoundError:
        return None

def build_and_run_device_query():
    # Change the directory to /usr/local/cuda/samples/1_Utilities/deviceQuery
    cuda_dir = "/usr/local/cuda/samples/1_Utilities/deviceQuery"
    os.chdir(cuda_dir)

    try:
        # Run 'make' command to build the deviceQuery binary
        subprocess.run(["make"], check=True)

        # Run the deviceQuery command
        result = subprocess.run(["./deviceQuery"], stdout=subprocess.PIPE, universal_newlines=True, check=True)

        # Get the output
        output = result.stdout

        # Find the line that contains the device name
        device_line = [line for line in output.splitlines() if "Device 0:" in line]
        if device_line:
            device_name = device_line[0].split(": ")[1].strip('"')
            return device_name
        else:
            return "Device not found in the output."
    except FileNotFoundError:
        return "deviceQuery or make command not found in the specified CUDA directory."