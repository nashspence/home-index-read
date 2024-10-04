from tika import parser, config
import json
import asyncio

tika_mimes = json.loads(config.getMimeTypes())

NAME = "tika"
FIELD_NAME = "tikaVersion"
MAX_WORKERS = 24
VERSION = 1

def does_support_mime(mime):
    archive_mimes = {
        'application/zip', 'application/x-tar', 'application/gzip',
        'application/x-bzip2', 'application/x-7z-compressed', 'application/x-rar-compressed'
    }
    
    if mime in tika_mimes:
        if (mime.startswith("text/") or mime.startswith("application/")) and mime not in archive_mimes:
            return True
    return False    

async def init():
    return

async def cleanup():
    return
    
async def get_fields(file_path, mime):
    parsed = await asyncio.to_thread(parser.from_file, file_path)
    return { "text": parsed.get("content", "") }