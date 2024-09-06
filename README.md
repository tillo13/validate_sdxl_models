
# Model Validation Scripts for SDXL10 and SDXL15

Welcome to the repository for validating SDXL10 and SDXL15 models. This repository contains scripts to validate a list of models fetched from Hugging Face based on their likes. The scripts generate images using these models and log the performance metrics such as success, error message, duration, and the folder size. The scripts also clean up the cache to ensure efficient storage usage.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
  - [validate_sdxl10_models.py](#validate_sdxl10_modelspy)
  - [validate_sdxl15_models.py](#validate_sdxl15_modelspy)
- [Results](#results)
- [Notes](#notes)

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.x
- `pip` (Python package installer)
- CUDA-enabled GPU (for better performance)

## Installation

1. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

   Before running this command, create a `requirements.txt` file in the repository root and include the following lines:
    ```txt
    requests
    opencv-python-headless
    torch
    torchvision
    diffusers
    huggingface-hub
    Pillow
    numpy
    insightface
    ```

2. (Optional) Download the IP-Adapter Checkpoint manually if not already downloaded:
    ```sh
    wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin -O checkpoints/ip_adapter_faceid_sd15.bin
    ```

## Usage

1. **Run the SDXL10 Model Validation Script:**
    ```sh
    python validate_sdxl10_models.py
    ```

2. **Run the SDXL15 Model Validation Script:**
    ```sh
    python validate_sdxl15_models.py
    ```

## File Descriptions

### validate_sdxl10_models.py

This script validates SDXL10 models. It does the following:

1. Fetches the top 1000 models from Hugging Face based on likes.
2. Downloads necessary checkpoints.
3. Loads a face analysis model to detect faces in the input image.
4. Tests each model by generating images and logging the performance metrics.
5. Deletes the cache to free up space.

#### Key Variables:
- `input_image_path`: Path to the input image.
- `output_dir`: Directory to save generated images.
- `csv_file`: CSV file to log the results.
- `models_url`: URL to fetch models from Hugging Face.

#### Key Functions:
- `download_checkpoints()`: Downloads model checkpoints.
- `fetch_models(url)`: Fetches the models from the given URL.
- `test_model(base_model_path, controlnet, input_image_path)`: Tests a single model and logs the performance metrics.

### validate_sdxl15_models.py

This script validates SDXL15 models. It follows a similar workflow to `validate_sdxl10_models.py` but is designed for SDXL15 models. 

#### Key Variables:
- `API_URL`: API endpoint to fetch the list of models.
- `input_image_path`: Path to the input image.
- `output_dir`: Directory to save generated images.
- `csv_file`: CSV file to log the results.
- `ip_ckpt_url`: URL to download the IP-Adapter checkpoint.

#### Key Functions:
- `fetch_models_sorted_by_likes(filter_tag, limit)`: Fetches models from the Hugging Face API based on specific criteria.
- `download_file(url, filename)`: Downloads a file if it doesn't exist locally.
- `test_model(base_model_path, vae, input_image_path)`: Tests a single model and logs the performance metrics.

## Results

The results are logged in CSV files (`model_test_results_sdxl10.csv` and `model_test_results_sdxl15.csv`) with the following columns:

- Model Path
- Success (True/False)
- Error (If any)
- Duration (Time taken for the generation in seconds)
- Folder Size (Cache folder size in GB for `validate_sdxl10_models.py`)

## Notes

- Ensure to verify the paths and URLs provided in the scripts.
- Large models and datasets may require substantial disk space and time.
- Both scripts clean up their respective cache directories after processing each model to maintain disk space efficiency.

By following the steps and the detailed descriptions above, you should be able to set up and run model validation for SDXL10 and SDXL15 models effectively.