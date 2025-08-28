import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
from io import BytesIO


class TableDetector:
    """
    A class for detecting tables in images
    using a pre-trained DETR model.
    """

    def __init__(self, model_name="TahaDouaji/detr-doc-table-detection", threshold=0.9):
        """
        Initialize the table detector with a pre-trained model, based on the documentation.

        :param model_name: HuggingFace model name to load
        :param threshold: Minimum confidence score to accept a prediction
        """
        # Processor handles preprocessing (inputs) and postprocessing (outputs)
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        # Load the object detection model
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        # Confidence threshold for filtering predictions
        self.threshold = threshold

    def predict(self, image_path: str):
        """
        Detect tables in an image.

        :param image_path: Path to the local image
        :return: List of dictionaries containing bounding boxes and confidence scores
        """
        try:
            # --- Load the image ---
            if image_path.startswith("http"):
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

            # --- Preprocessing ---
            inputs = self.processor(images=image, return_tensors="pt")

            # --- Inference ---
            with torch.no_grad():
                outputs = self.model(**inputs)

            # --- Post-processing ---
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]

            tables = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                print(f"DEBUG: score={score.item():.3f}, label={label.item()}, box={box.tolist()}")

                # Keep only predictions above the confidence threshold
                if score > self.threshold:
                    tables.append({
                        "score": score.item(),
                        "box": [round(x, 2) for x in box.tolist()]  # Rounded box coordinates
                    })

            if not tables:
                raise ValueError("No table detected")

            return tables

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")