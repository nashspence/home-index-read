# region "debugpy"


import os

if str(os.environ.get("DEBUG", "False")) == "True":
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
import gc
import subprocess
import mimetypes

from pathlib import Path
from wand.image import Image as WandImage
from home_index_module import run_server


# endregion
# region "config"


VERSION = 1
NAME = os.environ.get("NAME", "read")

LANGUAGES = str(os.environ.get("LANGUAGES", "en")).split(",")
MODEL_STORAGE_DIRECTORY = os.environ.get("MODEL_STORAGE_DIRECTORY", "/easyocr")
WORKERS = os.environ.get("WORKERS", 1)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 6)
GPU = str(os.environ.get("GPU", torch.cuda.is_available())) == "True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(
    os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
)


# endregion
# region "read images"


def read_images(file_path, max_dimension=1920):
    images = []
    with WandImage(filename=file_path, resolution=300) as img:
        img.format = "png"
        for frame in img.sequence:
            with WandImage(image=frame, resolution=300) as single_frame:
                single_frame.format = "png"
                width, height = single_frame.width, single_frame.height
                if max(width, height) > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int((max_dimension / width) * height)
                    else:
                        new_height = max_dimension
                        new_width = int((max_dimension / height) * width)
                    single_frame.resize(new_width, new_height)
                images.append(single_frame.make_blob())
    return images


# endregion
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

    del reader
    gc.collect()
    torch.cuda.empty_cache()


# endregion
# region "check/run"


def get_supported_formats():
    result = subprocess.run(
        ["identify", "-list", "format"], capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    supported = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("-") and not line.startswith("Format"):
            parts = line.split()
            if len(parts) >= 2:
                ext = parts[0].lower().rstrip("*")
                modes = parts[2]
                if "r" in modes:
                    supported.add(ext)
    return supported


SUPPORTED_FORMATS = get_supported_formats()


def get_extensions_from_mime(mime_type):
    return mimetypes.guess_all_extensions(mime_type)


def can_wand_open(mime_type):
    extensions = get_extensions_from_mime(mime_type)
    for ext in extensions:
        if ext.lstrip(".").lower() in SUPPORTED_FORMATS:
            return True
    return False


def check(file_path, document, metadata_dir_path):
    version_path = metadata_dir_path / "version.json"
    version = None

    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)

    if version and version.get("version") == VERSION:
        return False

    if document["type"].startswith("video/") or document["type"].startswith("audio/"):
        return False

    return can_wand_open(document["type"])


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    textbox_path = metadata_dir_path / "textboxes_per_image.json"
    plaintext_path = metadata_dir_path / "plaintext.txt"

    def attempt(batch_size=BATCH_SIZE):
        textboxes_per_image = []
        for image in read_images(file_path):
            textboxes_per_image.append(
                reader.readtext(
                    image, workers=WORKERS, batch_size=batch_size, paragraph=True
                )
            )
            gc.collect()
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

    exception = None
    try:
        attempt()
    except Exception as e:
        if str(e).startswith("CUDA out of memory."):
            try:
                logging.warning("CUDA out of memory. Retrying with BATCH_SIZE=1.")
                attempt(1)
            except Exception as e:
                exception = e
                logging.exception("failed")
        else:
            exception = e
            logging.exception("failed")

    version = {"version": VERSION}
    if exception:
        version["exception"] = str(exception)
    with open(version_path, "w") as file:
        json.dump(version, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(NAME, hello, check, run, load, unload)
