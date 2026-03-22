from django.urls import path
from .views import DetectClothingView, ListImagesView, DeleteImageView

urlpatterns = [
    path('detect/', DetectClothingView.as_view(), name='detect_clothing'),
    path('images/', ListImagesView.as_view(), name='list_images'),
    path('images/<int:pk>/', DeleteImageView.as_view(), name='delete_image'),
]
