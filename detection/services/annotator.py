import os
import uuid
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

# Define a color map for different clothing categories
# Colors are in RGB format
COLOR_MAP: Dict[str, tuple[int, int, int]] = {
    "shirt": (0, 255, 0),        # Green
    "pants": (0, 0, 255),        # Blue
    "shoes": (255, 165, 0),      # Orange
    "dress": (255, 0, 255),      # Magenta
    "hat": (255, 255, 0),        # Yellow
    "jacket": (0, 255, 255),     # Cyan
    "skirt": (128, 0, 128),      # Purple
    "sunglasses": (128, 128, 0), # Olive
    "bag": (0, 128, 128),        # Teal
}

# Default color for unknown labels
DEFAULT_COLOR: tuple[int, int, int] = (255, 0, 0) # Red

def annotate_image(image_path: str, detections: List[Dict[str, Any]]) -> str:
    """
    Annotate an image with bounding boxes and labels for each detection.
    
    Args:
        image_path: The full path to the original image.
        detections: A list of dicts, where each dict has:
                    - label: str
                    - confidence: float
                    - bbox: {"x1": int, "y1": int, "x2": int, "y2": int}
                    
    Returns:
        The full path to the annotated image, or the original image path if no detections.
    """
    if not detections:
        return image_path
        
    try:
        # 1. Open the image using Pillow
        with Image.open(image_path) as img:
            # Convert to RGB if it's not already (e.g., if it's RGBA or grayscale)
            # This ensures we can draw colored boxes properly
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            draw = ImageDraw.Draw(img)
            
            # Optionally, try to load a default font, otherwise use Pilar's built-in default
            try:
                # Try to load a generic truetype font if available
                # Pillow typically needs access to the system fonts
                font = ImageFont.truetype("arial.ttf", size=16)
            except IOError:
                # Fallback to the default bitmap font
                font = ImageFont.load_default()
            
            # 2. For each detection, draw the bounding box and label
            for det in detections:
                label = det.get("label", "unknown")
                confidence = det.get("confidence", 0.0)
                bbox = det.get("bbox", {})
                
                # Extract coordinates
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                x2 = bbox.get("x2", 0)
                y2 = bbox.get("y2", 0)
                
                # Get the color for this label
                color = COLOR_MAP.get(label.lower(), DEFAULT_COLOR)
                
                # Draw the bounding box (colored rectangle)
                # width parameter makes the line thicker
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Format the text: "{label} {confidence:.0%}"
                text = f"{label} {confidence:.0%}"
                
                # Calculate text size to draw a background rectangle for better readability
                try:
                    # Pillow 10+ syntax
                    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                    text_width = right - left
                    text_height = bottom - top
                except AttributeError:
                    # Older Pillow version syntax
                    text_width, text_height = draw.textsize(text, font=font)
                
                # Draw a filled rectangle behind the text for better visibility
                # Position it just above the bounding box top-left corner
                text_bg_x1 = x1
                text_bg_y1 = max(0, y1 - text_height - 4) # Ensure it doesn't go off the top edge
                text_bg_x2 = x1 + text_width + 4
                text_bg_y2 = text_bg_y1 + text_height + 4
                
                draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
                
                # Draw the text label (black text on colored background)
                draw.text((text_bg_x1 + 2, text_bg_y1 + 2), text, fill=(0, 0, 0), font=font)
            
            # 3. Save the annotated image to a temp file in the media/ folder
            # Extract the original filename
            original_filename = os.path.basename(image_path)
            
            # Generate a new filename with UUID: annotated_{uuid4}_{original_filename}
            new_filename = f"annotated_{uuid.uuid4()}_{original_filename}"
            
            # Determine the media directory path (based on the original image's directory or the api app structure)
            # Assuming the normal structure is workspace/detection/api/media/
            # Let's find the 'api' directory to construct the right path to 'media'
            api_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            media_dir = os.path.join(api_dir, "media")
            
            # Create media directory if it doesn't exist
            os.makedirs(media_dir, exist_ok=True)
            
            # Full path for the annotated image
            annotated_image_path = os.path.join(media_dir, new_filename)
            
            # Save the image
            img.save(annotated_image_path)
            
            # 4. Return the full path of the saved annotated image
            return annotated_image_path
            
    except Exception as e:
        # If any error occurs during drawing or saving, log it and return the original path
        # In a real app, you might want to log this error properly
        print(f"Error annotating image: {str(e)}")
        return image_path
