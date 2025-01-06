# region "debugpy"


import os
import debugpy

debugpy.listen(("0.0.0.0", 5678))

if str(os.environ.get("WAIT_FOR_DEBUGPY_CLIENT", "False")) == "True":
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
    print("Debugger attached.")
    debugpy.breakpoint()


# endregion
# region "import"


import io
import json
import logging
import numpy as np
import torch
import cv2

from PIL import Image
from wand.image import Image as WandImage
from home_index_module import run_server


# endregion
# region "config"


VERSION = 1
NAME = os.environ.get("NAME", "read")
LANGUAGE = os.environ.get("LANGUAGE", "en")
PYTORCH_DOWNLOAD_ROOT = os.environ.get("PYTORCH_DOWNLOAD_ROOT", "/root/.cache")
WORKERS = os.environ.get("WORKERS", 1)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 36)
GPU = str(os.environ.get("GPU", torch.cuda.is_available())) == "True"


# endregion


def read_images(file_path):
    images = []
    with WandImage(filename=file_path) as img:
        img.format = "png"
        for frame in img.sequence:
            with WandImage(image=frame) as single_frame:
                single_frame.format = "png"
                blob = single_frame.make_blob()
                pillow_image = Image.open(io.BytesIO(blob)).convert("RGB")
                images.append(cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR))
    return images


# region "hello"


def hello():
    return {
        "name": NAME,
        "version": VERSION,
        "filterable_attributes": [f"{NAME}.text"],
        "sortable_attributes": [],
    }


# endregion
# region "load/unload"


reader = None


def load():
    global reader
    import easyocr

    reader = easyocr.Reader(
        [LANGUAGE],
        gpu=GPU,
        model_storage_directory=PYTORCH_DOWNLOAD_ROOT,
        download_enabled=True,
    )


def unload():
    global reader
    import gc

    del reader
    gc.collect()
    torch.cuda.empty_cache()


# endregion
# region "check/run"


def check(file_path, document, metadata_dir_path):
    version_path = metadata_dir_path / "version.json"
    version = None
    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)
    if version and version.get("version") == VERSION:
        return False
    try:
        with WandImage(filename=file_path) as img:
            return True
    except:
        return False


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    textbox_path = metadata_dir_path / "textbox_data.json"
    plaintext_path = metadata_dir_path / "plaintext.txt"

    try:
        textbox_datas = [
            reader.readtext(
                image,
                workers=WORKERS,
                batch_size=BATCH_SIZE,
                paragraph=True
            )
            for image in read_images(file_path)
        ]

        plaintext = " ".join(
            textbox[1] for textbox_data in textbox_datas for textbox in textbox_data
        )

        document[NAME] = {}
        document[NAME]["text"] = plaintext

        with open(plaintext_path, "w") as file:
            file.write(plaintext)

        with open(textbox_path, "w") as file:
            json.dump(
                textbox_datas,
                file,
                indent=4,
                default=lambda o: (
                    int(o)
                    if isinstance(o, np.integer)
                    else float(o) if isinstance(o, np.floating) else str(o)
                ),
            )
    except Exception as e:
        logging.exception("failed")

    with open(version_path, "w") as file:
        json.dump({"version": VERSION}, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(hello, check, run, load, unload)
