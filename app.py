import os
import json
import re
import pandas as pd
import numpy as np
import cv2
import gradio as gr
from paddleocr import PaddleOCR
import joblib
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from datetime import datetime
import tempfile
from PIL import Image

# Import our modular components
from utils.ocr_processor import OCRProcessor
from utils.ml_classifier import EnhancedReceiptClassifier
from utils.extractors import extract_dates_from_text, extract_pin_from_text

class ReceiptProcessorApp:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.ml_classifier = EnhancedReceiptClassifier()
        self.setup_models()
    
    def setup_models(self):
        """Initialize ML models"""
        try:
            # Try to load models from local path
            model_paths = {
                'total': 'models/total_classifier.pkl',
                'vat': 'models/vat_classifier.pkl'
            }
            
            # If models don't exist locally, they'll use regex fallback
            self.ml_classifier.load_models(
                model_paths['total'],
                model_paths['vat'] if os.path.exists(model_paths['vat']) else None
            )
            print("✅ Models initialized")
        except Exception as e:
            print(f"⚠️ Model loading warning: {e}")
            print("ℹ️ Using regex fallback methods")

    def process_receipt(self, image):
        """Process a single receipt image"""
        try:
            # Convert Gradio image to temporary file
            if isinstance(image, str):
                image_path = image
            else:
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    image_path = tmp_file.name

            print(f"🔍 Processing image: {image_path}")

            # Step 1: OCR Processing
            with tempfile.TemporaryDirectory() as temp_dir:
                json_path = self.ocr_processor.process_image(image_path, temp_dir)
                
                if not json_path or not os.path.exists(json_path):
                    return {
                        "Total Amount": "❌ OCR failed",
                        "VAT Amount": "N/A",
                        "Date": "N/A", 
                        "PIN": "N/A",
                        "Status": "OCR processing failed"
                    }

                # Step 2: Load and process JSON data
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Step 3: Extract all fields
                result = self.ml_classifier.predict_all_fields(json_data)
                
                # Format results for display
                total_display = f"KES {result['total_amount']:,.2f}" if result['total_amount'] else "❌ Not found"
                vat_display = f"KES {result['vat_amount']:,.2f}" if result['vat_amount'] else "⏸️ Not detected"
                date_display = result['date'] if result['date'] else "⏸️ Not detected"
                pin_display = result['pin'] if result['pin'] else "⏸️ Not detected"
                
                status = "✅ Success" if result['total_amount'] else "⚠️ Partial success"

                # Clean up temporary image file if created
                if 'tmp_file' in locals():
                    os.unlink(image_path)

                return {
                    "Total Amount": total_display,
                    "VAT Amount": vat_display,
                    "Date": date_display,
                    "PIN": pin_display,
                    "Status": status
                }

        except Exception as e:
            print(f"❌ Processing error: {e}")
            # Clean up on error
            if 'tmp_file' in locals():
                try:
                    os.unlink(image_path)
                except:
                    pass
                    
            return {
                "Total Amount": f"❌ Error: {str(e)}",
                "VAT Amount": "N/A",
                "Date": "N/A",
                "PIN": "N/A", 
                "Status": f"Processing failed: {str(e)}"
            }

def create_interface():
    """Create Gradio interface"""
    processor = ReceiptProcessorApp()
    
    # Example images for demo
    example_images = []
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        example_images = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Receipt Processor") as demo:
        gr.Markdown("""
        # 🧾 Receipt Processor with VAT Extraction
        
        Upload a receipt image to extract:
        - **Total Amount** 💰
        - **VAT Amount** 🧾  
        - **Transaction Date** 📅
        - **Business PIN** 🔑
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Receipt Image",
                    sources=["upload"],
                    height=300
                )
                
                process_btn = gr.Button("🔄 Process Receipt", variant="primary", size="lg")
                
                gr.Examples(
                    examples=example_images[:4] if example_images else [],
                    inputs=image_input,
                    label="Try example receipts:"
                )
            
            with gr.Column():
                total_output = gr.Textbox(label="Total Amount", interactive=False)
                vat_output = gr.Textbox(label="VAT Amount", interactive=False)
                date_output = gr.Textbox(label="Date", interactive=False)
                pin_output = gr.Textbox(label="PIN", interactive=False)
                status_output = gr.Textbox(label="Status", interactive=False)
        
        # Processing info
        with gr.Accordion("ℹ️ How it works", open=False):
            gr.Markdown("""
            **Processing Pipeline:**
            1. **OCR Text Extraction** - Uses PaddleOCR to detect and read text from your receipt
            2. **Amount Detection** - Machine learning identifies the total amount with regex fallback
            3. **VAT Extraction** - Specialized pattern matching for tax amounts
            4. **Date & PIN Extraction** - Regex patterns for common date formats and PIN numbers
            
            **Supported Formats:** JPG, PNG, JPEG
            **Best Results:** Clear, well-lit receipt photos with horizontal text
            """)
        
        # Connect button to processing function
        process_btn.click(
            fn=processor.process_receipt,
            inputs=image_input,
            outputs=[total_output, vat_output, date_output, pin_output, status_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
