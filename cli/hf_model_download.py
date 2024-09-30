import os
from dotenv import load_dotenv
import argparse
from huggingface_hub import login, snapshot_download
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
HUGGINGFACE_ACCESS_TOKEN = os.getenv('HUGGINGFACE_ACCESS_TOKEN')

login(token=HUGGINGFACE_ACCESS_TOKEN)

def download_model(model_name):
    try:
        if 'onnx' in model_name:
            snapshot_download(
                model_name,
                revision='main',
                ignore_patterns=['*.git*', '*README.md'],
                local_dir=os.path.join(DATA_PATH, model_name)
            )
        else:
            model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
            model.save(os.path.join(DATA_PATH, model_name))
    except Exception as e:
        raise e
    
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help='download model from huggingface',
        required=True,
    )

    args = parser.parse_args()
    download_model(args.model)

if __name__ == '__main__':
    run()