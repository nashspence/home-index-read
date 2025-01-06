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
import rawpy
import torch
import pdf2image

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
BATCH_SIZE = os.environ.get("BATCH_SIZE", 16)
GPU = str(os.environ.get("GPU", torch.cuda.is_available())) == "True"


PDF_MIME_TYPES = {"application/pdf"}

RAW_MIME_TYPES = {
    "image/x-adobe-dng",
    "image/x-canon-cr2",
    "image/x-canon-crw",
    "image/x-nikon-nef",
    "image/x-sony-arw",
    "image/x-panasonic-raw",
    "image/x-olympus-orf",
    "image/x-fuji-raf",
    "image/x-sigma-x3f",
    "image/x-pentax-pef",
    "image/x-samsung-srw",
    "image/x-raw",
}

VECTOR_MIME_TYPES = {
    "image/svg+xml",
    "image/x-eps",
    "application/postscript",
    "application/eps",
    "image/vnd.adobe.photoshop",
    "application/vnd.adobe.photoshop",
    "application/x-photoshop",
    "application/photoshop",
    "image/vnd.adobe.illustrator",
    "application/vnd.adobe.illustrator",
    "application/illustrator",
    "application/x-illustrator",
}

STANDARD_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/tiff",
    "image/bmp",
    "image/webp",
}

SUPPORTED_MIME_TYPES = (
    PDF_MIME_TYPES | RAW_MIME_TYPES | VECTOR_MIME_TYPES | STANDARD_IMAGE_MIME_TYPES
)


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
    if not document["type"] in SUPPORTED_MIME_TYPES:
        return False
    version_path = metadata_dir_path / "version.json"
    version = None
    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)
    if version and version.get("version") == VERSION:
        return False
    return True


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    textbox_path = metadata_dir_path / "textbox_array.json"
    plaintext_path = metadata_dir_path / "plaintext.txt"

    images = []
    mime_type = document.get("type", "")
    if mime_type in STANDARD_IMAGE_MIME_TYPES:
        images.append(Image.open(file_path))
    elif mime_type in RAW_MIME_TYPES:
        with rawpy.imread(file_path) as raw:
            images.append(Image.fromarray(raw.postprocess()))
    elif mime_type in VECTOR_MIME_TYPES:
        with WandImage(filename=file_path) as img:
            img.format = "png"
            images.append(
                [
                    Image.open(io.BytesIO(img.sequence[i].make_blob()))
                    for i in range(len(img.sequence))
                ]
            )
    elif mime_type in PDF_MIME_TYPES:
        images = images.append(pdf2image.convert_from_path(file_path))

    textbox_array = reader.readtext_batched(
        [np.array(image) for image in images], workers=WORKERS, batch_size=BATCH_SIZE
    )

    plaintext = " ".join(item[1] for item in textbox_array)

    document[NAME] = {}
    document[NAME]["text"] = plaintext

    with open(plaintext_path, "w") as file:
        file.write(plaintext)

    with open(textbox_path, "w") as file:
        json.dump(
            textbox_array,
            file,
            indent=4,
            default=lambda o: (
                int(o)
                if isinstance(o, np.integer)
                else float(o) if isinstance(o, np.floating) else str(o)
            ),
        )

    with open(version_path, "w") as file:
        json.dump({"version": VERSION}, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(hello, check, run, load, unload)
