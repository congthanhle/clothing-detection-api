import os
import datetime
import uuid
import logging
import shutil
from pathlib import Path
from django.conf import settings
from detection.models import DetectionResult
from supabase import create_client, Client

logger = logging.getLogger(__name__)

supabase: Client = None
bucket_name = os.environ.get("BUCKET", "detections")
try:
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        supabase = create_client(url, key)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")

class StorageError(Exception):
    """Custom exception raised for storage errors."""
    pass

def upload_image(local_path: str) -> str:
    """
    Upload the image to Supabase Storage.
    Returns the public URL of the uploaded file.
    Raises StorageError on failure.
    """
    try:
        if not supabase:
            raise StorageError("Supabase client is not initialized.")
            
        filename = os.path.basename(local_path)
        date_str = datetime.date.today().isoformat()
        unique_id = uuid.uuid4()
        remote_filename = f"annotated/{date_str}/{unique_id}_{filename}"
        
        with open(local_path, "rb") as f:
            ext = filename.split('.')[-1].lower()
            content_type = "image/png" if ext == "png" else f"image/{ext}"
            content_type = "image/jpeg" if ext in ("jpg", "jpeg") else content_type

            res = supabase.storage.from_(bucket_name).upload(
                path=remote_filename, 
                file=f,
                file_options={"content-type": content_type, "upsert": "false"}
            )
            
        public_url = supabase.storage.from_(bucket_name).get_public_url(remote_filename)
        logger.info(f"Successfully uploaded image to Supabase. URL: {public_url}")
        return public_url

    except Exception as e:
        logger.error(f"Failed to save annotated image {local_path} to Supabase: {e}")
        raise StorageError(f"Failed to save annotated image: {e}")

def save_detection_record(
    original_path: str,
    annotated_url: str,
    detections: list[dict]
) -> dict:
    """
    Insert a detection result row into the local database.
    Returns the full inserted row as a dict.
    Raises StorageError on failure.
    """
    try:
        rel_path = ""
        if original_path and os.path.exists(original_path):
            try:
                rel_path = os.path.relpath(original_path, settings.MEDIA_ROOT)
                # Ensure the path uses forward slashes
                rel_path = rel_path.replace('\\', '/')
            except ValueError:
                rel_path = os.path.basename(original_path)

        record = DetectionResult.objects.create(
            original_image=rel_path,
            annotated_image_url=annotated_url,
            bounding_boxes=detections
        )
        
        logger.info(f"Saved DB record id={record.id} for file={rel_path}")
        
        return {
            "id": record.id,
            "original_filename": rel_path,
            "annotated_image_url": annotated_url,
            "detection_json": detections
        }
        
    except Exception as e:
        logger.error(f"Failed to save detection record: {e}")
        raise StorageError(f"Failed to save detection record: {e}")

def get_detection_record(record_id: str) -> dict:
    """
    Fetch a single detection_results row by id.
    Returns the row as a dict, or raises StorageError if not found.
    """
    try:
        record = DetectionResult.objects.get(id=record_id)
        return {
            "id": record.id,
            "original_filename": record.original_image.name,
            "annotated_image_url": record.annotated_image_url,
            "detection_json": record.bounding_boxes
        }
    except DetectionResult.DoesNotExist:
        raise StorageError(f"Record {record_id} not found.")
    except Exception as e:
        logger.error(f"Failed to fetch detection record {record_id}: {e}")
        raise StorageError(f"Failed to fetch detection record {record_id}: {e}")

def delete_local_file(local_path: str) -> None:
    """
    Delete the local temp file after an upload/processing succeeds.
    """
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except Exception as e:
        logger.error(f"Failed to delete local temp file '{local_path}': {e}")

def delete_image_record(record_id: str) -> bool:
    """
    Delete a DetectionResult record and its associated image from Supabase Storage.
    """
    try:
        record = DetectionResult.objects.get(id=record_id)
        
        if record.annotated_image_url and supabase:
            try:
                bucket_prefix = f"/object/public/{bucket_name}/"
                if bucket_prefix in record.annotated_image_url:
                    path_in_bucket = record.annotated_image_url.split(bucket_prefix)[1]
                    supabase.storage.from_(bucket_name).remove([path_in_bucket])
                    logger.info(f"Deleted {path_in_bucket} from Supabase Storage")
            except Exception as e:
                logger.warning(f"Failed to delete file from Supabase Storage: {e}")
                
        record.delete()
        return True
    except DetectionResult.DoesNotExist:
        raise StorageError(f"Record {record_id} not found.")
    except Exception as e:
        logger.error(f"Failed to delete record {record_id}: {e}")
        raise StorageError(f"Failed to delete record: {e}")
