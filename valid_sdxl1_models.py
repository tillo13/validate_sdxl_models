import os
import requests
import cv2
import torch
import time
import csv
import shutil
from datetime import datetime
from PIL import Image
from pathlib import Path
from diffusers import ControlNetModel
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from huggingface_hub import hf_hub_download
import numpy as np

# Paths and directories
input_image_path = "andy.jpg"
output_dir = "generated_images_sdxl10"
csv_file = "model_test_results_sdxl10.csv"
models_url = "https://huggingface.co/api/models?pipeline_tag=text-to-image&sort=likes&limit=1000"
KNOWN_WORKING_MODEL = 'RunDiffusion/Juggernaut-X-v10'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create output and checkpoint directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

# Checkpoints to download
checkpoints = {
    "controlnet_config": ("InstantX/InstantID", "ControlNetModel/config.json"),
    "controlnet_model": ("InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors"),
    "ip_adapter": ("InstantX/InstantID", "ip-adapter.bin")
}

# Function to download model files
def download_checkpoints():
    for fname, (repo, fpath) in checkpoints.items():
        local_path = Path(f"./checkpoints/{fpath}")
        if not local_path.exists():
            try:
                hf_hub_download(repo_id=repo, filename=fpath, local_dir="./checkpoints")
                print(f"Downloaded {fpath}")
            except Exception as e:
                print(f"Error downloading {fpath}: {e}")

# Download checkpoints
download_checkpoints()

# Fetch models from Hugging Face API
def fetch_models(url):
    response = requests.get(url)
    response.raise_for_status()
    models_data = response.json()
    model_paths = [model['modelId'] for model in models_data]
    return model_paths

# Load face analysis model
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_folder_size_in_gb(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)

def get_cache_dir(model_path):
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', 'models--' + model_path.replace('/', '--'))

def get_processed_models(csv_file):
    processed_models = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) > 0:
                    processed_models.add(row[0])
    return processed_models

def convert_from_cv2_to_image(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def test_model(base_model_path, controlnet, input_image_path):
    start_time = time.time()
    try:
        # Import the pipeline from your own module
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        face_adapter_path = "./checkpoints/ip-adapter.bin"
        pipe.load_ip_adapter_instantid(face_adapter_path)

        image = cv2.imread(input_image_path)
        faces = app.get(image)
        if len(faces) == 0:
            return False, "No faces detected", 0, 0

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
        face_image = convert_from_cv2_to_image(face_image)  # Convert to PIL Image

        prompt = "a person"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

        prompt = [prompt] * faceid_embeds.size(0)  # make batch sizes the same
        negative_prompt = [negative_prompt] * faceid_embeds.size(0)

        generator = torch.Generator(device=device).manual_seed(0)

        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=faceid_embeds,
            image=face_image,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator
        ).images

        duration = time.time() - start_time
        if images:
            timestamp = datetime.now().strftime("%H%M%S")
            output_filename = f"{os.path.basename(input_image_path).split('.')[0]}_{base_model_path.replace('/', '_')}_{timestamp}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            images[0].save(output_filepath)
            cache_dir = get_cache_dir(base_model_path)
            folder_size_gb = get_folder_size_in_gb(cache_dir) if os.path.exists(cache_dir) else 0
            return True, "", duration, folder_size_gb
        else:
            return False, "Image generation failed", duration, 0
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error in model {base_model_path}: {str(e)}")
        return False, str(e), duration, 0

def delete_cache_directory(model_path):
    try:
        cache_dir = get_cache_dir(model_path)
        print(f"Deleting cache for {model_path} at {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception as cache_error:
        print(f"Error deleting cache for {model_path}: {cache_error}")

def main():
    # First, test the known working model
    controlnet = ControlNetModel.from_pretrained('./checkpoints/ControlNetModel', torch_dtype=torch_dtype).to(device)

    print(f"**** TESTING KNOWN WORKING MODEL: {KNOWN_WORKING_MODEL} ****")
    success, error, duration, folder_size_gb = test_model(KNOWN_WORKING_MODEL, controlnet, input_image_path)
    print(f"Writing to CSV - Model: {KNOWN_WORKING_MODEL}, Success: {success}, Error: {error}, Duration: {duration}s, Folder Size: {folder_size_gb:.2f} GB")

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(csv_file) == 0:
            writer.writerow(["Model Path", "Success", "Error", "Duration (s)", "Folder Size (GB)"])

        writer.writerow([KNOWN_WORKING_MODEL, success, error, duration, folder_size_gb])
        file.flush()

    if not success:
        print("Known working model failed. Stopping further tests.")
        return

    # Proceed with the top 1000 models from the Hugging Face API
    model_paths = fetch_models(models_url)
    total_models = len(model_paths)
    print(f"**** FETCHED {total_models} MODELS ****")

    processed_models = get_processed_models(csv_file)
    print(f"**** FOUND {len(processed_models)} PROCESSED MODELS IN CSV ****")

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(csv_file) == 0:
            writer.writerow(["Model Path", "Success", "Error", "Duration (s)", "Folder Size (GB)"])

        for index, model_path in enumerate(model_paths, start=1):
            if model_path in processed_models:
                print(f"**** MODEL {model_path} ALREADY PROCESSED, SKIPPING ****")
                continue
            print(f"**** TESTING MODEL {model_path}: {index} OUT OF {total_models} ****")
            success, error, duration, folder_size_gb = test_model(model_path, controlnet, input_image_path)
            print(f"Writing to CSV - Model: {model_path}, Success: {success}, Error: {error}, Duration: {duration}s, Folder Size: {folder_size_gb:.2f} GB")
            writer.writerow([model_path, success, error, duration, folder_size_gb])
            file.flush()
            print(f"Model: {model_path}, Success: {success}, Error: {error}, Duration: {duration}s, Folder Size: {folder_size_gb:.2f} GB")

            delete_cache_directory(model_path)

if __name__ == "__main__":
    main()

print("\n===== TESTING COMPLETED =====")