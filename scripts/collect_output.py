from pathlib import Path
import shutil
from PIL import Image

script_dir = Path(__file__).resolve().parent
proj_dir = script_dir.parent


config = {
    "orig": [True, "GT", 0],
    "cubic": [False, "cubic", 0],
    "frvsr": [False, "frvsr", 32],
    "esrgan": [False, "esrgan", 0],
    "frvsresr": [False, "frvsresrgan", 224],
    "frvsresrgan": [False, "frvsresrgan", 256]
}

crop_config = {
    "UDM10_000": (780, 450, 80),
    "UDM10_003": (50, 170, 200),
    "UDM10_007": (100, 480, 240),
}

def get_orig_filenum(dataset: str, index: int):
    return f"{index:04}.png"

def copy(config_name: str, dataset: str, video_name: str):
    is_orig, train_name, frame_idx = config[config_name]
    if is_orig:
        input_dir = proj_dir / ".vsr" / "datasets" / dataset
        if dataset == "UDM10":
            input_dir = input_dir / "GT"
        elif dataset == "vid4":
            input_dir = input_dir / "original"
        input_dir = input_dir / video_name
        input_path = input_dir / get_orig_filenum(dataset, frame_idx)
    else:
        input_dir = proj_dir / "Results" / train_name / dataset / video_name
        input_path =input_dir / f"{video_name}_id0000_{frame_idx:04}.png"

    output_dir = proj_dir / "final_results" / f"{dataset}_{video_name}"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{config_name}.png"

    # print(input_path, output_path)
    shutil.copy(str(input_path), str(output_path))


def main():
    dataset = "UDM10"
    video_list = list(map(lambda x: f"{x:03}", list(range(10))))
    for config_name in config.keys():
        for video_name in video_list:
            copy(config_name, dataset, video_name)

    output_dir = proj_dir / "final_results"
    for key, c in crop_config.items():
        for config_name in config.keys():
            img = Image.open(output_dir / key / f"{config_name}.png")
            x, y, len = c
            img = img.crop([x, y, x + len, y + len])
            img.save(output_dir / key / f"{config_name}_crop.png")



if __name__ == '__main__':
    main()
