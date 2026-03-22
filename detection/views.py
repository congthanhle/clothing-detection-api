import os
import uuid
import logging
from pathlib import Path
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from .serializers import UploadImageSerializer, DetectionResultSerializer
from .services.detector import ClothingDetector, mock_detect, DetectionError
from .services.annotator import annotate_image
from .services.storage import upload_image, save_detection_record, delete_local_file, delete_image_record, StorageError
from .models import DetectionResult

logger = logging.getLogger(__name__)

class DetectClothingView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        original_path = None
        annotated_path = None

        logger.info("Step 1 START: Validate incoming file")
        serializer = UploadImageSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Step 1 END: Validation errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = serializer.validated_data['image']
        original_name = uploaded_file.name
        logger.info("Step 1 END: Validation successful")

        logger.info("Step 2 START: Save original image temporarily")
        uploads_dir = Path(settings.MEDIA_ROOT) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        original_path = str(uploads_dir / f"{uuid.uuid4()}_{original_name}")
        
        with open(original_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        logger.info(f"Step 2 END: Saved to {original_path}")

        try:
            logger.info("Step 3 START: Run detection")
            if getattr(settings, "USE_MOCK_DETECTOR", False):
                detections = mock_detect(original_path)
            else:
                detections = ClothingDetector().detect(original_path)
            logger.info("Step 3 END: Detection complete")

            logger.info("Step 4 START: Annotate image")
            annotated_path = annotate_image(original_path, detections)
            logger.info("Step 4 END: Annotate image complete")

            logger.info("Step 5 START: Upload annotated image to Supabase Storage")
            annotated_url = upload_image(annotated_path)
            logger.info(f"Step 5 END: Uploaded to {annotated_url}")

            logger.info("Step 6 START: Save record to Supabase DB")
            record_id = None
            try:
                record = save_detection_record(original_path, annotated_url, detections)
                record_id = str(record["id"])
            except StorageError as se:
                logger.warning(f"StorageError saving record: {se}")
            logger.info("Step 6 END: Save record complete")

            logger.info("Step 8 START: Return success response")
            response_data = {
                "annotated_image_url": annotated_url,
                "detections": detections,
                "record_id": record_id
            }
            logger.info("Step 8 END: Response data prepared")

        except DetectionError as de:
            logger.error(f"DetectionError during processing: {de}")
            return Response({"error": str(de)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except StorageError as se:
            logger.error(f"StorageError during processing: {se}")
            return Response({"error": str(se)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Unexpected Exception during processing: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            logger.info("Step 7 START: Cleanup local temp files")
            # original_path is kept as part of the detection record.
            if annotated_path:
                delete_local_file(annotated_path)
            logger.info("Step 7 END: Cleanup complete")

        return Response(response_data, status=status.HTTP_200_OK)

class ListImagesView(APIView):
    def get(self, request, *args, **kwargs):
        records = DetectionResult.objects.all().order_by('-created_at')
        serializer = DetectionResultSerializer(records, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class DeleteImageView(APIView):
    def delete(self, request, pk, *args, **kwargs):
        try:
            delete_image_record(pk)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except StorageError as se:
            if "not found" in str(se).lower():
                return Response({"error": "Record not found"}, status=status.HTTP_404_NOT_FOUND)
            return Response({"error": str(se)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
