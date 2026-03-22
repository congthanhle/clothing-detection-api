from rest_framework import serializers
from .models import DetectionResult

class DetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionResult
        fields = ['id', 'original_image', 'annotated_image_url', 'bounding_boxes', 'created_at']
class DetectionResponseSerializer(serializers.Serializer):
    annotated_image_url = serializers.CharField()
    detections = serializers.ListField()
    record_id = serializers.CharField()

class UploadImageSerializer(serializers.Serializer):
    image = serializers.ImageField(max_length=None, allow_empty_file=False)

    def validate_image(self, value):
        # Limit image size to 10MB (10 * 1024 * 1024 bytes)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image size must be under 10MB.")
        
        # Validate extensions to be jpg/png/webp
        ext = value.name.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'webp']:
            raise serializers.ValidationError("Unsupported file extension. Allowed: jpg, jpeg, png, webp.")
            
        return value
