from functools import lru_cache
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import os
import argparse
import numpy as np
import glob
import logging
from typing import Generator, List
from PIL import Image
import cv2
import tensorflow as tf, keras
from huggingface_hub import hf_hub_download
import csv
load_model_hf = tf.keras.models.load_model
from tqdm import tqdm
import json

gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0, "No GPUs available"
IMAGE_SIZE = 448
INTERPOLATION = cv2.INTER_AREA
REPOSITORY = "SmilingWolf/wd-v1-4-moat-tagger-v2" #moat or etc 
DTYPE = np.float16
# global params, models, general_tags, character_tags, rating_tags
model:tf.keras.models.Model = None
general_tags:list|None = None
character_tags:list|None = None

logging_path = 'detect.log'
logging.basicConfig(filename=logging_path, level=logging.ERROR)
if gpus:
  # set memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def convert_images(image_paths: List[str]):
    """
    Converts images to PIL.Image.Image
    Returns list of (image_path, image) which was successful
    """
    success_paths = []
    result = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            success_paths.append(image_path)
            result.append(image)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            else:
                logging.exception(f"Exception occured: {e} for {image_path}")
            continue
    return success_paths, result

def active_preprocessor(imgs, batch_size=1):
    """
    Generator, but actively preprocesses images and return as prepared
    :param imgs: image paths
    :param batch_size: batch size
    :yields: (image_paths, images)
    """
    pooling_executor = ThreadPoolExecutor(max_workers=1)
    futures = []
    # split imgs into minibatch
    divmod_result = divmod(len(imgs), batch_size)
    batch_count = divmod_result[0]
    if divmod_result[1] != 0:
        batch_count += 1
    minibatches = []
    for i in range(batch_count):
        minibatches.append(imgs[i*batch_size:min((i+1)*batch_size, len(imgs))])
    for minibatch in minibatches:
        # handle Image.open in thread
        futures.append(pooling_executor.submit(convert_images, minibatch))
    for future in futures:
        yield future.result()

def submit_yolo(model, minibatch):
    """
    Submits YOLO model with minibatch
    Minibatch is list of (image_path, image)
    """
    result = model(minibatch[1])
    return minibatch[0], result

def detect(imgs, cuda_device=0, model='yolov8n.pt', batch_size=-1,stream:bool=False, max_infer_size=640, conf_threshold=0.3, iou_threshold=0.5):
    """
    Detects images with YOLO model
    Returns list of (image_path, result) which was successful
    """
    try:
        if not len(imgs):
            return []
        thread_pool = ThreadPoolExecutor(max_workers=1) # 1 thread per device, allows asynchronous execution and preprocessing
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        model = YOLO(model) #YOLO('yolov8n-face.pt') # for face only
        model.conf = conf_threshold
        model.iou = iou_threshold
        model.imgsz = max_infer_size
        result_list = []
        if batch_size == -1:
            batch_size = len(imgs)
        else:
            batch_size = min(batch_size, len(imgs))
        # split imgs into minibatch
        divmod_result = divmod(len(imgs), batch_size)
        batch_count = divmod_result[0]
        if divmod_result[1] != 0:
            batch_count += 1
        minibatches_provider = active_preprocessor(imgs, batch_size)
        futures = []
        for minibatch in tqdm(minibatches_provider, desc=f'minibatch with device {cuda_device}', total=batch_count):
            # print(f"handling {minibatch}") # debug once
            #result_list.extend(model(minibatch))
            # send to thread and get future, do not block
            # verbose=False
            futures.append(thread_pool.submit(submit_yolo, model, minibatch))
        # wait for all futures
        for future in tqdm(futures, desc=f'Waiting for futures with device {cuda_device}'):
            try:
                if stream:
                    # wait for each future
                    #print("Yielding")
                    result = future.result()
                    yield from zip(result[0], result[1])
                else:
                    # wait for each future
                    result = future.result()
                    for image_path, result in zip(result[0], result[1]):
                        result_list.append((image_path, result))
            except Exception as execption:
                if isinstance(execption, KeyboardInterrupt):
                    raise execption
                logging.error(f'Exception occured: {execption}')
                continue
        if not stream:
            return result_list
    except Exception as execption:
        logging.error(f'Exception occured: {execption}')
        raise execption

def crop_by_person(image: Image.Image, box_xyxy: list):
    """
    Crop image by person's box
    :param image: image to crop
    :param box_xyxy: person's box
    :return: cropped images as list
    """
    cropped_images = []
    for box in box_xyxy:
        # convert tensor to list
        box = box.tolist()
        cropped_images.append(image.crop(box))
    return cropped_images

def save_cropped_images(image: Image.Image, box_xyxy: list, original_filepath:str, save_dir: str):
    """
    Save cropped images to save_dir
    :param image: image to crop
    :param box_xyxy: person's box as list
    :param original_filepath: original image's filepath
    :param save_dir: directory to save cropped images
    :return: None
    """
    #print(box_xyxy)
    filename_without_ext = os.path.splitext(os.path.basename(original_filepath))[0] # pure filename without extension
    cropped_images = crop_by_person(image, box_xyxy)
    if not cropped_images:
        return # debugging
    for i, cropped_image in enumerate(cropped_images):
        cropped_image.save(os.path.join(save_dir, f'{filename_without_ext}_{i}.jpg'))

def detect_and_save_cropped_images(image_paths: List[str], save_dir: str, cuda_device: int = 0, model:str = 'yolov8n.pt', idx: int = 0, batch_size: int = -1, max_infer_size: int = 640, conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    """
    Detect person and save cropped images
    :param image_path: image path
    :param save_dir: directory to save cropped images
    :param cuda_device: cuda device number
    :param model: model name, 'yolov8n.pt' or 'yolov8n-face.pt'
    :param idx: index of box to use as person box
    :param batch_size: minibatch size, -1 means all images at once
    :param max_infer_size: max inference size
    :param conf_threshold: confidence threshold
    :param iou_threshold: iou threshold
    :return: None
    
    # xyxy[idx] is used for person box as index 0
    """
    if len(image_paths) == 0:
        return
    result_container = detect(image_paths, cuda_device, model, batch_size, stream=True, max_infer_size=max_infer_size, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    for path, r in result_container:
        #print(f'handling {path}')
        #print(r.boxes)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            logging.error(f"Exception occured: {e}")
            continue
        where_idx = r.boxes.cls.cpu().numpy() == idx # person classes # [True, False, True, ...]
        xyxy = r.boxes.xyxy[where_idx] # [tensor([x1, y1, x2, y2]), tensor([x1, y1, x2, y2]), ...]
        if len(xyxy) != 2: # we only allow 2 persons to be detected
            logging.error(f"Expected 2 persons, got {len(xyxy)} for {path}")
            continue
        save_cropped_images(image, xyxy, path, save_dir)

def execute_yolo(cuda_devices:str, image_path:str, recursive:bool, save_dir:str, batch_size:int, model:str = 'yolov8n.pt',
        max_infer_size:int = 640, conf_threshold:float = 0.3, iou_threshold:float = 0.5):
    """
    Executes detection with YOLO model and save cropped images
    :param cuda_devices: cuda device numbers, comma separated
    :param image_path: image path
    :param recursive: recursive
    :param save_dir: directory to save cropped images
    :param batch_size: minibatch size, -1 means all images at once
    :param model: model name, 'yolov8n.pt' or 'yolov8n-face.pt'. Recommended : person_detect_plus_v1.1_m.pt by DeepGHS
    :param max_infer_size: max inference size
    :param conf_threshold: confidence threshold
    :param iou_threshold: iou threshold
    
    The images are saved in save_dir
    """
    image_exts = ['jpg', 'jpeg', 'png', 'webp']
    image_paths = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if recursive:
        # use os.walk
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.split('.')[-1] in image_exts:
                    image_paths.append(os.path.join(root, file))
    else:
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(image_path, f'*.{ext}'), recursive=False))
    print(f'found {len(image_paths)} images')
    # detect and save cropped images
    available_cuda_devices = cuda_devices.split(',')
    available_cuda_devices = [int(cuda_device) for cuda_device in available_cuda_devices]
    # split image_paths into cuda_devices using numpy.array_split
    image_paths_split = np.array_split(image_paths, len(available_cuda_devices))
    # debug with raw execution
    #for cuda_device, image_paths in zip(available_cuda_devices, image_paths_split):
    #    detect_and_save_cropped_images(image_paths, save_dir, cuda_device, batch_size=batch_size)
    #return
    try:
        with ProcessPoolExecutor(max_workers=len(available_cuda_devices)) as executor:
            results = []
            for cuda_device, image_paths in zip(available_cuda_devices, image_paths_split):
                results.append(executor.submit(detect_and_save_cropped_images, image_paths, save_dir, cuda_device, batch_size=batch_size,model=model,
                                max_infer_size=max_infer_size, conf_threshold=conf_threshold, iou_threshold=iou_threshold
                                ))
            for result in results:
                try:
                    result.result()
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    logging.error(f'Exception occured: {e}')
                    continue
    except KeyboardInterrupt:
        executor.shutdown(wait=False)
        print('KeyboardInterrupt')
        exit(1)
  


def read_tags(base_path):
    """
    Reads tags from selected_tags.csv, and stores them in global variables
    base_path: base path to model (str)
    return: None
    """
    global general_tags, character_tags
    if general_tags is not None and character_tags is not None:
        return None
    with open(os.path.join(base_path, 'selected_tags.csv'), "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        tags = list(reader)
        header = tags.pop(0)
        tags = tags[1:]
    assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"header is not correct for {base_path} selected_tags.csv"
    # if category is 0, general, 4, character, else ignore
    general_tags = [tag[1] for tag in tags if tag[2] == '0']
    character_tags = [tag[1] for tag in tags if tag[2] == '4']
    return None

def preprocess_image(image:Image.Image) -> np.ndarray:
    global IMAGE_SIZE, INTERPOLATION
    # handle RGBA
    assert isinstance(image, Image.Image), f"Expected image to be Image.Image, got {type(image)} with value {image}"
    if image.mode == "RGBA":
        # paste on white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        image = background
    image = np.array(image)
    image = image[:, :, ::-1].copy() # RGB to BGR
    # pad to square image
    target_size = [max(image.shape)] * 2
    # pad with 255 to make it white
    image_padded = 255 * np.ones((target_size[0], target_size[1], 3), dtype=np.uint8)
    dw = int((target_size[0] - image.shape[1]) / 2)
    dh = int((target_size[1] - image.shape[0]) / 2)
    image_padded[dh:image.shape[0]+dh, dw:image.shape[1]+dw, :] = image
    image = image_padded
    # assert
    assert image.shape[0] == image.shape[1]
    
    # resize
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=INTERPOLATION)
    image = image.astype(DTYPE)
    return image

def download_model(repo_dir: str = REPOSITORY, save_dir: str = "./", force_download: bool = False):
    # tagger follows following files
    print("Downloading model")
    FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
    SUB_DIR = "variables"
    SUB_DIR_FILES = [f"{SUB_DIR}.data-00000-of-00001", f"{SUB_DIR}.index"]
    if os.path.exists(save_dir) and not force_download:
        return os.path.abspath(save_dir)
    # download
    for file in FILES:
        print(f"Downloading {file}")
        hf_hub_download(repo_dir, file, cache_dir = save_dir, force_download = force_download, force_filename = file)
    for file in SUB_DIR_FILES:
        print(f"Downloading {file}")
        hf_hub_download(repo_dir, (SUB_DIR+'/'+ file), cache_dir = os.path.join(save_dir, SUB_DIR), force_download = force_download, force_filename = file)
    return os.path.abspath(save_dir)

# check if model is already loaded in port 5050, if it is, use api
def check_model_loaded():
    """
    Checks if model is loaded in port_number
    port_number: port number to check (int)
    ip: ip address to check (str) default: localhost
    return: True if model is loaded, False otherwise (bool)
    """
    return model is not None

    
def load_model(model_path: str = "./models", force_download: bool = False):
    """
    Loads model from model_path
    model_path: path to model (str)
    force_download: force download model (bool)
    port_number: port number to load model (int)
    return: None
    """
    if check_model_loaded():
        return None
    if not model_path:
        raise ValueError("model_path is None")
    if (not os.path.exists(model_path)) or force_download:
        download_model(REPOSITORY, model_path, force_download = force_download)
    # load model
    global model
    print("Loading model")
    # precisions
    model = load_model_hf(model_path)
    return None

def predict_tags(prob_list:np.ndarray, threshold=0.5, model_path:str="./") -> List[str]:
    """
    Predicts tags from prob_list
    prob_list: list of probabilities, first 4 are ratings, rest are tags
    threshold: threshold for tags (float)
    model_path: path to model (str)
    return: list of tags (list of str)
    """
    global model, general_tags, character_tags
    probs = np.array(prob_list)
    #ratings = probs[:4] # first 4 are ratings
    #rating_index = np.argmax(ratings)
    tags = probs[4:] # rest are tags
    if general_tags is None or character_tags is None:
        read_tags(model_path)
    assert general_tags is not None and character_tags is not None, "general_tags and character_tags are not loaded"
    result = []
    for i, p in enumerate(tags):
        if i < len(general_tags) and p > threshold:
            tag_name = general_tags[i]
            # replace _ with space
            tag_name = tag_name.replace("_", " ")
            result.append(tag_name)
    return result

def predict_image(image: np.ndarray, model_path: str = "./") -> List[str]:
    """
    Predicts image from image
    image: image to predict (np.ndarray)
    model_path: path to model (str)
    return: list of tags (list of str)
    """
    global model
    if model is None:
        load_model(model_path)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    probs = model.predict(image)[0]
    return predict_tags(probs, model_path=model_path)

def predict_images_batch(images: Generator, model_path: str = "./", batch_size = 16, minibatch_size = 16,total:int=-1, action:callable = None, threadexecutor:ThreadPoolExecutor = None) -> List[List[str]]:
    """
    Predicts images from images
    images: images to predict (list of np.ndarray)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    global model
    if model is None:
        load_model(model_path)
    assert images is not None, "images is None"
    #images = [preprocess_image(image) for image in images]
    results = []
    full, partial = divmod(total, batch_size)
    total_batches = full + bool(partial)
    _total_done = 0
    for i in tqdm(range(total_batches), desc="GPU Batch"):
        batch = []
        paths = []
        for j in tqdm(range(batch_size), desc=f"Loading batch {i} over {total_batches}"):
            try:
                path, image = next(images)
                assert isinstance(image, Image.Image), f"Expected image to be Image.Image, got {type(image)} with value {image}"
                assert isinstance(path, str), f"Expected path to be str, got {type(path)} with value {path}"
                batch.append(image)
                paths.append(path)
            except StopIteration:
                break
        #batch = images[i*batch_size:(i+1)*batch_size]
        batch_processed = []
        for image in tqdm(batch, desc="Preprocessing", total=total, initial=_total_done):
            batch_processed.append(preprocess_image(image))
            _total_done += 1
        batch = np.array(batch_processed)
        # With proper handling, the later part can be threaded (GPU side, load next batch while predicting)
        threadexecutor.submit(threaded_job, batch, paths, minibatch_size, model_path, action)
        #threaded_job(batch, paths, minibatch_size, model_path, action)
    return results

def threaded_job(batch, paths, minibatch_size, model_path, action):
    try:
        probs = model.predict(batch, batch_size=minibatch_size)
        # move to cpu (tensorflow-gpu)
        # clear session
        keras.backend.clear_session()
        tags_batch = []
        for prob in probs:
            tags = predict_tags(prob, model_path=model_path)
            tags_batch.append(tags)
        del probs
        if action is not None:
            action(paths, tags_batch)
    except Exception as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            raise e
        return None

def handle_yield(image_path_list:List[str]):
    # yields image_path, image if image_path is valid and image loads
    for image_path in image_path_list:
        try:
            image = Image.open(image_path)
            yield image_path, image
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            continue

# for image paths, locally
def predict_images_from_path(image_paths: List[str], model_path: str = "./", action=None, batch_size=2048,minibatch_size=16) -> List[List[str]]:
    """
    Predicts images from image_paths
    image_paths: paths to images (list of str)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    # check if model is loaded
    if not check_model_loaded():
        global model
        if model is None:
            load_model(os.path.abspath(model_path))
    generator = handle_yield(image_paths)
    executor = ThreadPoolExecutor(max_workers=1)
    # batch size is 2048, minibatch size is 16 - batch size (RAM) / minibatch size (GPU)
    return predict_images_batch(generator, model_path=model_path, action=action, total=len(image_paths), batch_size=batch_size,minibatch_size = minibatch_size, threadexecutor=executor)

# using glob, get all images in a folder then request
def predict_local_path(path:str, recursive:bool=False, action:callable=None, max_items:int=0, batch_size=2048, minibatch_size=16) -> dict[str, List[str]]: # path: path to folder
    """
    Predicts images from path
    path: path to folder (str)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    # get all images in path
    import glob
    paths = []
    if not recursive:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            paths.extend(glob.glob(os.path.join(path, ext)))
    else:
        #os.walk
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".webp"):
                    paths.append(os.path.join(root, file))
    if max_items > 0:
        paths = paths[:max_items]
    print(f"Found {len(paths)} images")
    # post and get
    result = predict_images_from_path(paths, action=action, batch_size=batch_size, minibatch_size=minibatch_size)
    result_dict = {x[0]:x[1] for x in zip(paths, result)}
    return result_dict

def move_matching_items(paths:List[str], tags:List[List[str]]):
    """
    Moves images to matching tags
    This is an example of action which moves images to matching tags
    """
    for path, tag in zip(paths, tags):
        if any("futa" in y for y in tag):
            target_path = os.path.join(r"D:\interrogate\matches", os.path.basename(path))
            if os.path.exists(os.path.join(r"D:\interrogate\matches", os.path.basename(path))):
                _i = 0
                while os.path.exists(target_path):
                    target_path = os.path.join(r"D:\interrogate\matches", f"{os.path.basename(path)}_{_i}")
                    _i += 1
            os.rename(path, target_path)
            print(f"Moved {path} to {target_path}")

def save_to_file(paths:List[str], tags:List[List[str]]):
    """
    Saves tags to file (same name as image, but txt)
    """
    for path, tag in zip(paths, tags):
        ext = os.path.splitext(path)[1]
        text_path = path.replace(ext, ".txt")
        if os.path.exists(text_path):
            continue
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"{tag}")
            
def relocate_main(original_image_path, segmented_image_path, save=False):
    original_general_tags = {}
    result_dict = {}
    # EXTRACT GENERAL TAGS FROM ORIGINAL IMAGES
    for filename in listdir_cached(original_image_path):
        original_image_name = os.path.splitext(filename)[0]
        # import pdb; pdb.set_trace()
        if filename.endswith('.txt') and not filename.endswith('_relocated.txt'): #skip relocated txt files
            image_name = os.path.splitext(filename)[0]
            tags_file_path = os.path.join(original_image_path, filename)
            original_general_tags[image_name] = extract_general_tags(tags_file_path)
    
            # PROCESS SEGMENTED IMAGES
            segmented_info = process_segmented_image(original_image_name, segmented_image_path)
            if not segmented_info:
                continue
            # SORT SEGMENTS BY HORIZONTAL CENTER AND GROUP TAGS
            relocated_tags_line = sort_and_group_tags(segmented_info, original_general_tags.get(image_name, []))
            result_dict[image_name] = relocated_tags_line
            # Save the relocated tags as new txt files
            original_tags_file_path = os.path.join(original_image_path, f'{image_name}.txt')
            save_path = os.path.join(original_image_path, f'{image_name}_relocated.txt')
            if save:
                save_relocated_tags(save_path, open(original_tags_file_path, 'r').readlines(), relocated_tags_line)
    return result_dict
                
  
def sort_and_group_tags(segmented_info, original_tags):
    # Sort segments by horizontal center
    sorted_segments = sorted(segmented_info.items(), key=lambda x: get_horizontal_center(x[1]['coords']) if x[1]['coords'] else float('inf'))
    
    # Group the general tags by segment and order them from left to right
    matched_tag_list = []
    for segment_id, segment in sorted_segments:
        if 'tags' in segment:
            common_tags = [tag for tag in original_tags if tag in segment['tags']]
            matched_tag_list.append(common_tags)
            
    flattened_matched_tags = [tag for sublist in matched_tag_list for tag in sublist]
    remaining_tags = [tag for tag in original_tags if tag not in flattened_matched_tags]
    # replace space in tag with '_'
    remaining_tags = [tag.replace(' ', '_') for tag in remaining_tags]
    
    # replace space in tag with '_' in matched_tag_list
    matched_tag_list = [[tag.replace(' ', '_') for tag in tags] for tags in matched_tag_list]
    
    relocated_tags = ' ||| '.join([' '.join(tags) for tags in matched_tag_list])
    relocated_tags = ' '.join(remaining_tags)+(' ||| ' + relocated_tags if relocated_tags else '')
    return relocated_tags
  
def normalize_tags(tags):
    """Normalizes tags by replacing underscores with spaces."""
    return [tag.replace('_', ' ') for tag in tags]  

def extract_general_tags(tags_file):   
    """Extracts general tags from the given tags file."""
    with open(tags_file, 'r', encoding='utf-8') as file:
        file_lines = file.readlines()
        # if GENERAL_TAGS: line exists, extract tags from it
        # else, everything is general tags
        if any("GENERAL TAGS:" in line for line in file_lines):
            for line in file_lines:
                if 'GENERAL TAGS:' in line:
                    general_tags = line.replace('GENERAL TAGS:', '').strip()
                    return normalize_tags(general_tags.split())
        else:
            return normalize_tags(" ".join(file_lines).split())


def process_coordinates(coords_file):
    """Extracts coordinates from the given coordinates file."""
    with open(coords_file, 'r') as file:
        coords = file.read().strip().split(' ')
        return tuple(map(float, coords))
    
@lru_cache(maxsize=128)
def listdir_cached(path):
    return os.listdir(path)

def process_segmented_image(original_image, segmented_image_path):
    """
    Processes the segmented image and returns the cropped images.
    Returns a dictionary with the following format:
    {
        'image_id_segment_id': {
            'coords': (x1, y1, x2, y2),
            'tags': ['tag1', 'tag2']
        }
    }
    The dictionary may be empty if no segmented images are found.
    """
    # while filename in segmented_image_path has 'original_image' in it:
    #   process the image
    segmented_info = {}
    for filename in listdir_cached(segmented_image_path):
        if not filename.startswith(original_image + '_'):
            continue
        base_name, extension = os.path.splitext(filename)
        image_id, segment_id = base_name.split('_')[:2]
        segment_key = f"{image_id}_{segment_id}"
        if segment_key not in segmented_info:
            segmented_info[segment_key] = {'coords': None, 'tags': []}
            
        if filename.endswith('_coords.txt'):
            coords = process_coordinates(os.path.join(segmented_image_path, filename))
            segmented_info[segment_key]['coords'] = coords

        elif filename.endswith('.txt') and not filename.endswith('_coords.txt'):
            tags = process_tags(os.path.join(segmented_image_path, filename))
            segmented_info[segment_key]['tags'] = tags
    
    return segmented_info


def process_tags(tags_file):
    with open(tags_file, 'r', encoding='utf-8') as file:
        tags = file.read().strip()
        tags = tags.strip("[]").replace("'", "").split(', ')
        return tags

def get_horizontal_center(coords):
    """Returns the horizontal center of the given coordinates."""
    x1, _, x2, _ = coords
    return (x1 + x2) / 2  

def save_relocated_tags(save_path, original_tag_lines, relocated_tag_lines):
    """Saves the relocated tags as a new txt file."""
    with open(save_path, 'w', encoding='utf-8') as file:
        if any("GENERAL TAGS:" in line for line in original_tag_lines):
            # find GENERAL TAGS: line and replace it with relocated tags
            for line in original_tag_lines:
                if 'GENERAL TAGS:' in line:
                    file.write('GENERAL TAGS: ' + relocated_tag_lines + '\n')
                else:
                    file.write(line)
        else:
            # replace all
            for line in relocated_tag_lines:
                file.write(line)

def save_to_json(original_image_path, json_path):
    """Saves the relocated tags and its image name as a new json file."""
    # save relocated txt file name and its image name as a json file
    json_data = []
    for filename in listdir_cached(original_image_path):
        if filename.endswith('_relocated.txt'):
            image_name = os.path.splitext(filename)[0].split('_')[0] 
            with open(os.path.join(original_image_path, filename), 'r') as f:
                tags = ""
                lines = f.readlines()
                if any("GENERAL TAGS:" in line for line in lines):
                    for line in lines:
                        if 'GENERAL TAGS:' in line:
                            tags = line.replace('GENERAL TAGS:', '').strip()
                else:
                    tags = " ".join(lines).strip()
            json_data.append({'image_name': image_name, 'tags': tags})
            
    json_file_path = os.path.join(json_path)
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-devices', type=str, default='0', help='cuda device numbers, comma separated')
    parser.add_argument('--image-path', type=str, required=True, help='image path')
    parser.add_argument('--recursive', action='store_true', help='recursive')
    parser.add_argument('--save-dir', type=str, required=True, help='directory to save cropped images')
    parser.add_argument('--batch-size', type=int, default=-1, help='minibatch size, -1 means all images at once')
    parser.add_argument('--yolo_model', type=str, default='person_detect_plus_v1.1_best_m.pt', help='model name, yolov8n.pt or yolov8n-face.pt')
    parser.add_argument('--force-download', action='store_true', help='force download model')
    parser.add_argument('--tagger-model', type=str, default='SmilingWolf/wd-v1-4-moat-tagger-v2', help='tagger model name')
    parser.add_argument('--model-path', type=str, default='./models', help='model path')
    parser.add_argument('--max-infer-size', type=int, default=640, help='max inference size')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--save-json', type=str, required=True, help='directory to save json file')
    args = parser.parse_args()
    
    REPOSITORY = args.tagger_model
    import torch
    assert torch.cuda.is_available(), "No CUDA available"
    execute_yolo(args.cuda_devices, args.image_path, args.recursive, args.save_dir, args.batch_size, args.yolo_model, args.max_infer_size, args.conf_threshold, args.iou_threshold)
    
    load_model(args.model_path, args.force_download) # Tagger
    # adjust by yourself
    predict_local_path(args.save_dir, recursive=True, action=save_to_file, max_items = 0, batch_size=256, minibatch_size=32).values() # ban tags   
    
    # relocate general tags
    result_dict = relocate_main(args.image_path, args.save_dir)
    
    with open(args.save_json, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
    # save imagename:tags to json file
    #save_to_json(args.image_path, args.save_json)
