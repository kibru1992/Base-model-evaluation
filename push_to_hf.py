import os
from huggingface_hub import HfApi, create_repo

def push_to_huggingface(repo_id, token, local_dir):
    api = HfApi(token=token)
    
    print("Verifying token...")
    try:
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Error: Token verification failed. Please check your HF_TOKEN.\nDetails: {e}")
        return

    print(f"Creating/Verifying repository: {repo_id}...")
    try:
        # We'll try creating it as a model repository first
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        if "403" in str(e):
            print(f"\n[!] Error 403: You don't have permission to create a repository under '{repo_id.split('/')[0]}'.")
            print("Please ensure your token has 'Write' or 'Repo Creation' permissions.")
            print("Go to: https://huggingface.co/settings/tokens and create a new token with 'Write' access.")
        else:
            print(f"Error during repository creation: {e}")
        return

    print(f"Uploading files from {local_dir} to {repo_id}...")
    files_to_upload = ["README.md", "evaluate_model.py"]
    
    for filename in files_to_upload:
        file_path = os.path.join(local_dir, filename)
        if os.path.exists(file_path):
            print(f"Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    token=token,
                    repo_type="model"
                )
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")
        else:
            print(f"File not found: {file_path}")

    print("\n[SUCCESS] Upload complete!")
    print(f"View your project here: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # It is recommended to set these as environment variables or use a .env file
    # If using a .env file, install python-dotenv and add: from dotenv import load_dotenv; load_dotenv()
    
    REPO_NAME = os.environ.get("HF_REPO_NAME", "evaluation-falcon3-1b-base")
    USER_NAME = os.environ.get("HF_USERNAME", "k4christ")
    HF_TOKEN = os.environ.get("HF_TOKEN") # DO NOT hardcode this for GitHub
    
    local_directory = r"C:\Users\kibru\Desktop\fatima"
    repository_id = f"{USER_NAME}/{REPO_NAME}"
    
    print("--- Hugging Face Upload Tool (Secure Mode) ---")
    
    if not HF_TOKEN:
        print("\n[!] Error: HF_TOKEN environment variable not found.")
        print("To fix this:")
        print("1. Create a .env file based on .env.example")
        print("2. Or run: $env:HF_TOKEN='your_token_here'; python push_to_hf.py")
    else:
        push_to_huggingface(repository_id, HF_TOKEN, local_directory)
