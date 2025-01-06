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

from PIL import Image
from pdf2image import convert_from_path
from wand.image import Image as WandImage
from home_index_module import run_server


# endregion
# region "config"

VERSION = 1
NAME = os.environ.get("NAME", "read")
LANGUAGE = os.environ.get("LANGUAGE", "en")
PYTORCH_DOWNLOAD_ROOT = os.environ.get("PYTORCH_DOWNLOAD_ROOT", "/app/data/pytorch")
WORKERS = os.environ.get("WORKERS", 1)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 4)
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
# region "read text"


reader = None


def get_textbox_array(image):
    global reader
    np_image = np.array(image)
    text = reader.readtext(np_image, workers=WORKERS, batch_size=BATCH_SIZE)
    return " ".join(text).strip()


def get_plaintext(data):
    if isinstance(data, list):
        return "\n\n".join(filter(None, (get_plaintext(item) for item in data)))
    elif isinstance(data, str):
        return data.strip()
    return ""


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
    global logging
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    textbox_path = metadata_dir_path / "textbox_array.json"
    plaintext_path = metadata_dir_path / "plaintext.txt"

    textbox_array = None

    mime_type = document.get("type", "")
    if mime_type in STANDARD_IMAGE_MIME_TYPES:
        textbox_array = get_textbox_array(Image.open(file_path))
    elif mime_type in RAW_MIME_TYPES:
        with rawpy.imread(file_path) as raw:
            rgb_image = raw.postprocess()
            image_pil = Image.fromarray(rgb_image)
            textbox_array = get_textbox_array(image_pil)
    elif mime_type in VECTOR_MIME_TYPES:
        with WandImage(filename=file_path, resolution=300) as img:
            img.format = "png"
            image_blobs = [
                img.sequence[i].make_blob() for i in range(len(img.sequence))
            ]
            images = [np.array(Image.open(io.BytesIO(blob))) for blob in image_blobs]
            texts_list = reader.readtext_batched(images, detail=0)
            textbox_array = "\n".join([" ".join(texts) for texts in texts_list])
    elif mime_type in PDF_MIME_TYPES:
        pages = convert_from_path(file_path, dpi=300)
        images = [np.array(page) for page in pages]
        texts_list = reader.readtext_batched(images, detail=0)
        textbox_array = "\n".join([" ".join(texts) for texts in texts_list])

    plaintext = get_plaintext(textbox_array)

    document[NAME] = {}
    document[NAME]["text"] = plaintext

    with open(plaintext_path, "w") as file:
        file.write(plaintext)

    with open(textbox_path, "w") as file:
        json.dump(textbox_array, file, indent=4)

    with open(version_path, "w") as file:
        json.dump({"version": VERSION}, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(hello, check, run, load, unload)
