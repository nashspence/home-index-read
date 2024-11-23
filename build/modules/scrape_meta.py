import time
import json
import tika
tika.TikaClientOnly = True
from tika import config

NAME = "scrape"
VERSION = 1
PATH = "/app/modules/scrape.py"
MAX_WORKERS = 16
FILTERABLE_FIELD_NAMES = [
    'altitude',
    'audio_bit_depth',
    'audio_bit_rate',
    'audio_channels',
    'audio_codec',
    'audio_sample_rate',
    'camera_lens_make',
    'camera_lens_model',
    'creation_date',
    'creation_date_precision'
    'creation_date_is_inferred',
    'creation_date_offset_seconds',
    'creation_date_offset_is_inferred',
    'creation_date_special_filepath_type',
    'device_make',
    'device_model',
    'duration',
    'height',
    'latitude',
    'longitude'
    'video_bit_rate',
    'video_codec',
    'video_frame_rate',
    'width'
]
SORTABLE_FIELD_NAMES = [
    'duration',
    'creation_date'
]

TIKA_MIMES = {}
for attempt in range(30):
    try:
        TIKA_MIMES = set(json.loads(config.getMimeTypes()))
    except:
        time.sleep(1 * attempt)
            
def inspect(pdoc, cdoc, fp, dir):
    version = None
    version_path = dir / f"{NAME}.json"
    if version_path.exists():
        with open(version_path, 'r') as file:
            version = json.load(file)  
    if version and version.get("file_path") == fp and version.get("version") == VERSION:
        return False
    return True