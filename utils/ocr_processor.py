import os
import json
from paddleocr import PaddleOCR

class OCRProcessor:
    def __init__(self):
        self.ocr = None
        self.initialize_ocr()

    def initialize_ocr(self):
        """Initialize PaddleOCR with optimized settings"""
        try:
            print("📖 Initializing PaddleOCR...")
            # Use lighter models for faster inference
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                rec_image_shape='3, 48, 320',
                det_limit_side_len=960,  # Limit image size for speed
                det_limit_type='max',
                use_dilation=False,  # Faster processing
                show_log=False
            )
            print("✅ OCR initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize OCR: {e}")
            self.ocr = None

    def process_image(self, image_path, output_folder):
        """Process single image with OCR"""
        if self.ocr is None:
            print("❌ OCR not initialized")
            return None

        try:
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)
            
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)

            # Extract text and polygons
            texts = []
            polys = []
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) == 2:
                        poly, (text, confidence) = line
                        if confidence > 0.5:  # Confidence threshold
                            texts.append(text)
                            polys.append(poly)

            # Save as JSON
            json_data = {
                "rec_texts": texts,
                "dt_polys": polys,
                "image_path": image_path
            }
            
            json_path = os.path.join(output_folder, "output.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            print(f"   ✅ OCR completed: {len(texts)} text elements found")
            return json_path

        except Exception as e:
            print(f"   ❌ OCR processing error: {e}")
            return None
