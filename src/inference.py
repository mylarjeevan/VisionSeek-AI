from ultralytics import YOLO
from pathlib import Path
import time
import torch
from src.config import load_config
from src.metadata_service import enrich_record, image_id_for
from src.utils import portable_relpath


class YOLOv11Inference:
    def __init__(self, model_name, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(self.device)

        config = load_config()
        self.conf_threshold = config["model"]["conf_threshold"]
        self.extensions = config["data"]["image_extension"]

    def process_image(self, image_path):
        started = time.time()
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
        )

        detection = []
        class_counts = {}

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                detection.append(
                    {
                        "class": cls,
                        "confidence": conf,
                        "bbox": bbox,
                        "count": 1,
                    }
                )
                class_counts[cls] = class_counts.get(cls, 0) + 1

        for det in detection:
            det["count"] = class_counts[det["class"]]

        elapsed = time.time() - started
        record = {
            "image_id": image_id_for(image_path),
            "image_path": portable_relpath(image_path),
            "filename": Path(image_path).name,
            "detections": detection,
            "total_objects": len(detection),
            "unique_class": list(class_counts.keys()),
            "class_counts": class_counts,
        }
        return enrich_record(record, model_used=self.model_name, elapsed=elapsed)

    def process_directory(self, directory):
        metadata = []
        patterns = [f"*{ext}" for ext in self.extensions]
        image_paths = []
        for pattern in patterns:
            image_paths.extend(Path(directory).glob(pattern))

        for img_path in image_paths:
            try:
                metadata.append(self.process_image(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        return metadata
