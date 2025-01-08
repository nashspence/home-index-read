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


import json
import logging
import torch

from wand.image import Image as WandImage
from home_index_module import run_server


# endregion
# region "config"


VERSION = 1
NAME = os.environ.get("NAME", "read")

LANGUAGES = str(os.environ.get("LANGUAGES", "en")).split(",")
MODEL_STORAGE_DIRECTORY = os.environ.get("MODEL_STORAGE_DIRECTORY", "/easyocr")
WORKERS = os.environ.get("WORKERS", 2)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 8)
GPU = str(os.environ.get("GPU", torch.cuda.is_available())) == "True"


# endregion


def read_images(file_path):
    images = []
    with WandImage(filename=file_path, resolution=300) as img:
        img.format = "png"
        for frame in img.sequence:
            with WandImage(image=frame, resolution=300) as single_frame:
                single_frame.format = "png"
                images.append(single_frame.make_blob())
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
        LANGUAGES,
        gpu=GPU,
        model_storage_directory=MODEL_STORAGE_DIRECTORY,
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
        with WandImage(filename=file_path):
            return True
    except Exception:
        return False


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    textbox_path = metadata_dir_path / "textboxes_per_image.json"
    plaintext_path = metadata_dir_path / "plaintext.txt"

    exception = None
    try:
        textboxes_per_image = []
        for image in read_images(file_path):
            textboxes_per_image.append(
                reader.readtext(
                    image, workers=WORKERS, batch_size=BATCH_SIZE, paragraph=True
                )
            )
            torch.cuda.empty_cache()

        plaintext = " ".join(
            textbox[1]
            for image_textboxes in textboxes_per_image
            for textbox in image_textboxes
        )

        document[NAME] = {}
        document[NAME]["text"] = plaintext

        with open(plaintext_path, "w") as file:
            file.write(plaintext)
        with open(textbox_path, "w") as file:
            json.dump(textboxes_per_image, file)
    except Exception as e:
        exception = e
        logging.exception("failed")

    with open(version_path, "w") as file:
        json.dump({"version": VERSION, "exception": str(exception)}, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(hello, check, run, load, unload)
