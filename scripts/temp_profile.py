from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from collect_output import config, get_orig_filenum


script_dir = Path(__file__).resolve().parent
proj_dir = script_dir.parent


def read_image(input_dir: Path, output_path: Path):
    path_list = sorted(list(input_dir.glob("*.png")))
    assert len(path_list) != 0
    first_path = path_list[0]
    base_img = Image.open(first_path)
    w = base_img.width
    h = base_img.height
    dst = Image.new('RGB', (w, h + len(path_list)))
    dst = Image.new('RGB', (w + len(path_list), h))
    dst.paste(base_img, (0, 0))
    for i, input_path in enumerate(path_list):
        img = Image.open(input_path)
        # img = img.crop([0, h // 2, w, h // 2 + 1])
        # # img = img.crop([0, h // 20 * 19, w, h // 20 * 19 + 1])
        # dst.paste(img, (0, h + i))

        img = img.crop([w // 2, 0, w // 2 + 1, h])
        dst.paste(img, (w + i, 0))
    dst.save(output_path)
    # img = Image.open(input_path)
    # print(img)
    # print(img.width)
    # print(img.height)

def new_read_image(config_name: str, dataset: str, video_name: str, cut_percent: float, vertical: bool):
    is_orig, train_name, frame_idx = config[config_name]
    if is_orig:
        input_dir = proj_dir / ".vsr" / "datasets" / dataset
        if dataset == "UDM10":
            input_dir = input_dir / "GT"
        elif dataset == "vid4":
            input_dir = input_dir / "original"
        input_dir = input_dir / video_name
        input_path_list = list(map(lambda x: input_dir / f"{x:04}.png", range(frame_idx, frame_idx + 32)))
    else:
        input_dir = proj_dir / "Results" / train_name / dataset / video_name
        input_path_list = list(map(lambda x: input_dir / f"{video_name}_id0000_{x:04}.png", range(frame_idx, frame_idx + 32)))

    base_img = Image.open(input_path_list[0])
    w = base_img.width
    h = base_img.height
    if vertical:
        dst = Image.new('RGB', (w, len(input_path_list)))
        cut_h = int(cut_percent * h)
        for i, input_path in enumerate(input_path_list):
            img = Image.open(input_path)
            img = img.crop([0, cut_h, w, cut_h + 1])
            dst.paste(img, (0, i))
    else:
        dst = Image.new('RGB', (len(input_path_list), h))
        cut_w = int(cut_percent * w)
        for i, input_path in enumerate(input_path_list):
            img = Image.open(input_path)
            img = img.crop([cut_w, 0, cut_w + 1, h])
            dst.paste(img, (i, 0))

    output_dir = proj_dir / "final_results" / f"{dataset}_{video_name}"
    output_dir.mkdir(exist_ok=True, parents=True)

    if vertical:
        output_path = output_dir / f"{config_name}_temp_v.png"
    else:
        output_path = output_dir / f"{config_name}_temp_h.png"

    dst.save(output_path)


def main():
    # output_dir = proj_dir / "results" / "temp_profile"
    # output_dir.mkdir(exist_ok=True, parents=True)

    # original_img_path = proj_dir / ".vsr" / "datasets" / "vid4" / "original" / "city"
    # old_img_path = proj_dir / "Results" / "frvsr" / "VID4" / "city"
    # new_img_path = proj_dir / "Results" / "frvsrnew" / "VID4" / "city"

    # original_img_path = proj_dir / ".vsr" / "datasets" / "UDM10" / "GT" / "007"
    # old_img_path = proj_dir / "Results" / "frvsr" / "UDM10" / "007"
    # new_img_path = proj_dir / "Results" / "frvsrnew" / "UDM10" / "007"
    # srgan_img_path = proj_dir / "Results" / "esrgan" / "UDM10" / "007"
    #
    # read_image(original_img_path, output_dir / "orig_temp_profile.png")
    # read_image(old_img_path, output_dir / "old_temp_profile.png")
    # read_image(new_img_path, output_dir / "new_temp_profile.png")
    # read_image(srgan_img_path, output_dir / "srgran_temp_profile.png")

    # setup_name = [
    #     "frvsr", "esrgan", "frvsresrgan"
    # ]

    # for setup in setup_name:
    #     img_path = proj_dir / "Results" / setup / "UDM10" / "007"
    #     read_image(img_path, output_dir / f"{setup}_temp_profile.png")

    temp_config = {
        "000": 0.5,
        "003": 0.5,
        "007": 0.5,
        "008": 0.5,
        "009": 0.5,
    }

    dataset = "UDM10"
    vertical = False
    for config_name in config.keys():
        for video_name in temp_config.keys():
            new_read_image(config_name, dataset, video_name, temp_config[video_name], vertical)


if __name__ == '__main__':
    main()


