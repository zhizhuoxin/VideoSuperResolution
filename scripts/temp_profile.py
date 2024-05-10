from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


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



def main():
    # original_img_path = proj_dir / ".vsr" / "datasets" / "vid4" / "original" / "city"
    # old_img_path = proj_dir / "Results" / "frvsr" / "VID4" / "city"
    # new_img_path = proj_dir / "Results" / "frvsrnew" / "VID4" / "city"

    original_img_path = proj_dir / ".vsr" / "datasets" / "UDM10" / "GT" / "007"
    old_img_path = proj_dir / "Results" / "frvsr" / "UDM10" / "007"
    new_img_path = proj_dir / "Results" / "frvsrnew" / "UDM10" / "007"
    srgan_img_path = proj_dir / "Results" / "esrgan" / "UDM10" / "007"

    output_dir = proj_dir / "results" / "temp_profile"
    output_dir.mkdir(exist_ok=True, parents=True)

    read_image(original_img_path, output_dir / "orig_temp_profile.png")
    read_image(old_img_path, output_dir / "old_temp_profile.png")
    read_image(new_img_path, output_dir / "new_temp_profile.png")
    read_image(srgan_img_path, output_dir / "srgran_temp_profile.png")


if __name__ == '__main__':
    main()


