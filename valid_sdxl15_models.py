import os
import requests
import cv2
import torch
import csv
import shutil
import time
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# API endpoint to fetch the list of models sorted by likes
API_URL = "https://huggingface.co/api/models"

# List of potential JSON configuration file names
CONFIG_FILES = ["model_index.json", "config.json"]

# Fetch models based on specific criteria
def fetch_models_sorted_by_likes(filter_tag="text-to-image", limit=1008):
    params = {
        "pipeline_tag": filter_tag,
        "sort": "likes",
        "direction": -1,  # Descending order
        "limit": limit
    }
    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    models = response.json()
    return [model['modelId'] for model in models]

# Get list of Hugging Face model paths to test
model_paths = fetch_models_sorted_by_likes()
total_models = len(model_paths)
print(f"Found {total_models} models sorted by likes to test.")

# VAE model path
vae_model_name = "stabilityai/sd-vae-ft-mse"

# Paths and directories
input_image_path = "incoming_images/brad.jpg"
output_dir = "generated_images_sd15"
os.makedirs(output_dir, exist_ok=True)

# CSV file for logging results
csv_file = "model_test_results_sd15.csv"

# Define URL and path for the IP-Adapter checkpoint
ip_ckpt_path = "./checkpoints/ip_adapter_faceid_sd15.bin"
ip_ckpt_url = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin?download=true"

# Helper function to download a file if it doesn't exist
def download_file(url, filename):
    if not os.path.isfile(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

# Download the IP-Adapter checkpoint if it doesn't exist
download_file(ip_ckpt_url, ip_ckpt_path)

# Load face analysis model
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)

# Function to get the cache directory for a given model path
def get_cache_dir(model_path):
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', 'models--' + model_path.replace('/', '--'))

# Function to find the correct config file URL for a given model path
def find_config_url(base_model_path):
    for config_file in CONFIG_FILES:
        url = f"https://huggingface.co/{base_model_path}/resolve/main/{config_file}"
        response = requests.head(url)
        if response.status_code == 200:  # Found the file
            return url
    raise FileNotFoundError(f"None of the config files ({', '.join(CONFIG_FILES)}) were found for model: {base_model_path}")

# Function to test image generation with a given model and VAE
def test_model(base_model_path, vae, input_image_path):
    start_time = time.time()
    try:
        config_url = find_config_url(base_model_path)

        # Attempt to load the model using the correct configuration
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            config=config_url,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        ).to(device)

        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
        ip_model = IPAdapterFaceID(pipe, ip_ckpt_path, device)

        # Load and process input image
        image = cv2.imread(input_image_path)
        faces = app.get(image)
        if len(faces) == 0:
            return False, "No faces detected", 0

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

        # Generate image with minimal parameters for testing
        prompt = "a person"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            width=1024,
            height=1024,
            num_inference_steps=75
        )

        duration = time.time() - start_time  # Calculate duration

        # Save generated image
        if images:
            output_filename = f"{base_model_path.replace('/', '_')}_test_output.png"
            output_filepath = os.path.join(output_dir, output_filename)
            images[0].save(output_filepath)
            return True, "", duration
        else:
            return False, "Image generation failed", duration
    except Exception as e:
        duration = time.time() - start_time  # Calculate duration even in case of error
        return False, str(e), duration

# Main function to test and log results
def main():
    vae = AutoencoderKL.from_pretrained(vae_model_name).to(dtype=torch.float16).to(device)

    tested_models = set()

    # Read the existing CSV file to avoid re-testing models
    if os.path.isfile(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                tested_models.add(row[0])

    for i, model_path in enumerate(model_paths, start=1):
        if model_path in tested_models:
            print(f"Skipping already tested model {model_path}")
            continue

        print(f"****** BEGINNING {i} of {total_models} MODEL {model_path} ******")
        success, error, duration = test_model(model_path, vae, input_image_path)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_path, success, error, duration])

        # Print the result of testing for the current model
        result_msg = f"Model: {model_path}, Success: {success}, Error: {error if error else 'None'}, Duration: {duration:.2f}s"
        print(result_msg)

        print(f"****** FINISHED {i} of {total_models} MODEL {model_path} ******\n")

        # Always try to delete the cache directory regardless of success or failure
        try:
            cache_dir = get_cache_dir(model_path)
            print(f"Deleting cache for {model_path} at {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as cache_error:
            print(f"Error deleting cache for {model_path}: {cache_error}")

if __name__ == "__main__":
    if not os.path.isfile(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model Path", "Success", "Error", "Duration (s)"])

    main()

print("\n===== TESTING COMPLETED =====")