import os
import tarfile
from huggingface_hub import hf_hub_download
import gdown

# Define datasets configuration mapping closely to Terramind examples
DATASETS = {
    "sen1floods11": {
        "target_dir": "sen1floods11_v1.1",
        "archive_name": "sen1floods11_v1_1.tar.gz",
        "hf_repo": "blumenstiel/Sen1Floods11",
        "gdrive_url": "https://google.com"
    },
    "burnscars": {
        "target_dir": "hls_burn_scars",
        "archive_name": "hls_burn_scars.tar.gz",
        "hf_repo": "blumenstiel/BurnScars",
        "gdrive_url": "https://google.com"
    }
}

def extract_tar(archive_path, extract_path="./"):
    """Safely extracts tar.gz archives using standard python libraries."""
    print(f"📦 Extracting {archive_path} to {extract_path}...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"✅ Extraction completed successfully.")
        # Clean up the archive file to save disk space
        os.remove(archive_path)
    except Exception as e:
        print(f"❌ Failed to extract {archive_path}: {e}")

def fetch_dataset(name, config):
    print(f"\n--- Processing Dataset: {name.upper()} ---")
    
    # Check if data directory already exists
    if os.path.isdir(config["target_dir"]):
        print(f"⏭️  Directory '{config['target_dir']}' already exists. Skipping download.")
        return

    # Strategy 1: Attempt Hugging Face Download
    try:
        print(f"🚀 Attempting Hugging Face Hub download from '{config['hf_repo']}'...")
        downloaded_file = hf_hub_download(
            repo_id=config["hf_repo"],
            filename=config["archive_name"],
            repo_type="dataset",
            local_dir="./"
        )
        print(f"📥 Successfully fetched via Hugging Face.")
        extract_tar(downloaded_file)
        return
    except Exception as hf_error:
        print(f"⚠️ Hugging Face fetch failed: {hf_error}")
        print("🔄 Falling back to Google Drive backup strategy...")

    # Strategy 2: Fallback to Google Drive Download
    try:
        print(f"🚀 Downloading via Gdown from storage mirror...")
        output_archive = config["archive_name"]
        gdown.download(config["gdrive_url"], output_archive, quiet=False)
        if os.path.exists(output_archive):
            print(f"📥 Successfully fetched via Google Drive.")
            extract_tar(output_archive)
        else:
            print(f"❌ Google Drive completed but target archive missing.")
    except Exception as gd_error:
        print(f"❌ Critical Error: Both download pipelines failed for {name}. Details: {gd_error}")

if __name__ == "__main__":
    # Execute loop across both workflows sequentially
    for dataset_name, dataset_config in DATASETS.items():
        fetch_dataset(dataset_name, dataset_config)
    print("\n🏁 All pipeline download checks complete.")
