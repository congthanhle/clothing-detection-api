from django.db import models

class DetectionResult(models.Model):
    original_image = models.ImageField(upload_to='uploads/')
    annotated_image_url = models.URLField(max_length=500, blank=True, null=True)
    bounding_boxes = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"DetectionResult {self.id} - {self.created_at}"
