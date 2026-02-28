from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="predictive_maintenance/deployment",     # folder containing all files to host streamlit app
    repo_id="sathish39893/predictive-maintenance",          # the target repo
    repo_type="space",                      # space as it need to host streamlit app
    path_in_repo="",                          # optional: subfolder path inside the repo
)
