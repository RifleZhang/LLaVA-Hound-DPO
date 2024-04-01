import os, sys, time, os.path as osp

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", False)
api = HfApi(token=HF_TOKEN)

def upload_file(repo_id, local_path, repo_path, repo_type, **kwargs):
    if not osp.isfile:
        raise(f"not file {local_path}")
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_type=repo_type,
        **kwargs
    )
    
def upload_folder(repo_id, local_folder, repo_path, repo_type, **kwargs):
    if not osp.isdir(local_folder):
        raise(f"not folder {local_folder}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_folder,
        path_in_repo=repo_path,
        repo_type=repo_type,
        **kwargs
    )

def download_file(repo_id, filename, local_dir, repo_type, subfolder=None, **kwargs):
    hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        subfolder=subfolder,
        local_dir=local_dir,
        token=HF_TOKEN,
        **kwargs
    )

def download_repo(repo_id, local_dir, repo_type, **kwargs):
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        token=HF_TOKEN,
        **kwargs
    )
