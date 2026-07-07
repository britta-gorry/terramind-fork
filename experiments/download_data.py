from huggingface_hub import hf_hub_download
hf_hub_download(
        repo_id='ibm-esa-geospatial/Examples',
        filename='S2L2A/Santiago.tif',
        repo_type='dataset',
        local_dir='examples/'
    )