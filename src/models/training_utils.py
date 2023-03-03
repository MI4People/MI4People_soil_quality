from pathlib import Path
from src.infrastructure.aws_infrastructure import get_s3_folder_content


def get_latest_weights(folder_path: str):
    if "s3" in folder_path:
        files = [Path(file) for file in get_s3_folder_content(folder_path)]
    else:
        files = [file for file in Path(folder_path).iterdir() if file.is_file()]
    epoch_and_weights = [(filename, filename.stem.split("_")[-1]) for filename in files]
    return max(epoch_and_weights, key=lambda x: x[1])[0]
