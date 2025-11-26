import os
import re
import numpy as np
import joblib
from .extractors import extract_dates_from_text, extract_pin_from_text

def normalize_y(poly):
    """Average vertical position of polygon"""
    return sum([p[1] for p in poly]) / len(poly)

def get_average_x(poly):
    """Get average X coordinate of a polygon"""
    return sum([p[0] for p in poly]) / len(poly)

def clean_amount(text):
    """Clean numeric amount from extracted text"""
    if not text:
        return None
    cleaned = re.sub(r'[^\d,.]', '', str(text))
    cleaned = cleaned.replace(',', '')
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
    cleaned = cleaned.rstrip('.')
    try:
        val = float(cleaned) if cleaned else None
        if val and 1 <= val <= 5000000:
            return val
        return None
    except:
        return None

def extract_all_amounts(layout_data):
    """Extract all potential amounts from the receipt"""
    amounts = []
    for text, poly, y_pos in layout_data:
        patterns = [r'\b\d+\.\d{2}\b', r'\b\d{3,}\b']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                amount = clean_amount(match)
                if amount and amount >= 10:
                    amounts.append({
                        'amount': amount, 'text': text, 'y_pos': y_pos, 'poly': poly
                    })
    return amounts

def extract_all_tax_amounts(layout_data):
    """Extract all potential tax amounts from the receipt"""
    tax_amounts = []
    for text, poly, y_pos in layout_data:
        patterns = [r'\b\d+\.\d{2}\b', r'\b\d{2,}\b']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                amount = clean_amount(match)
                if amount and amount >= 1:
                    tax_amounts.append({
                        'amount': amount, 'text': text, 'y_pos': y_pos, 'poly': poly
                    })
    return tax_amounts

class EnhancedReceiptClassifier:
    def __init__(self):
        self.total_model = None
        self.vat_model = None
        self.ml_confidence_threshold = 0.4

    def load_models(self, total_model_path, vat_model_path=None):
        """Load ML models"""
        try:
            # Load total model
            if os.path.exists(total_model_path):
                self.total_model = joblib.load(total_model_path)
                print("✅ Total model loaded")
            else:
                print("ℹ️ No total model found, using regex fallback")

            # Load VAT model
            if vat_model_path and os.path.exists(vat_model_path):
                self.vat_model = joblib.load(vat_model_path)
                print("✅ VAT model loaded")
            else:
                print("ℹ️ No VAT model found, using regex fallback")

            return True
        except Exception as e:
            print(f"⚠️ Model loading warning: {e}")
            return False

    def predict_all_fields(self, json_data):
        """Extract all fields from JSON data"""
        try:
            texts = json_data.get("rec_texts", [])
            polys = json_data.get("dt_polys", json_data.get("text_det_polys", []))

            if not texts or not polys:
                return {
                    'total_amount': None,
                    'vat_amount': None,
                    'date': None,
                    'pin': None
                }

            # Create layout data
            layout_data = []
            for txt, poly in zip(texts, polys):
                if txt and poly:
                    layout_data.append((txt.strip(), poly, normalize_y(poly)))
            layout_data.sort(key=lambda x: x[2])

            full_text = " ".join(txt for txt, _, _ in layout_data)

            # Extract fields
            total_amount = self.extract_total_with_regex(layout_data, full_text)
            date = extract_dates_from_text(full_text)
            pin = extract_pin_from_text(full_text)
            
            # Extract VAT
            all_amounts = extract_all_amounts(layout_data)
            vat_amount = self.extract_vat_with_regex(layout_data, full_text, all_amounts)

            return {
                'total_amount': total_amount,
                'vat_amount': vat_amount,
                'date': date,
                'pin': pin
            }

        except Exception as e:
            print(f"❌ Field extraction error: {e}")
            return {
                'total_amount': None,
                'vat_amount': None, 
                'date': None,
                'pin': None
            }

    def extract_total_with_regex(self, layout_data, full_text):
        """Extract total amount using regex patterns"""
        all_amounts = extract_all_amounts(layout_data)
        if not all_amounts:
            return None

        # Look for amounts near total keywords
        total_keywords = ["TOTAL", "AMOUNT DUE", "AMOUNT PAYABLE", "GRAND TOTAL", "BALANCE", "TOT"]

        for i, (text, poly, y_pos) in enumerate(layout_data):
            text_upper = text.upper()

            if any(keyword in text_upper for keyword in total_keywords):
                # Amount in same line
                line_amounts = [amt for amt in all_amounts if abs(amt['y_pos'] - y_pos) < 5]
                if line_amounts:
                    return max(line_amounts, key=lambda x: x['amount'])['amount']

                # Amount in next line
                if i + 1 < len(layout_data):
                    next_text, next_poly, next_y = layout_data[i + 1]
                    next_amounts = [amt for amt in all_amounts if abs(amt['y_pos'] - next_y) < 5]
                    if next_amounts:
                        return max(next_amounts, key=lambda x: x['amount'])['amount']

        # Fallback to largest amount in bottom half
        max_y = max(amt['y_pos'] for amt in all_amounts)
        bottom_amounts = [amt for amt in all_amounts if amt['y_pos'] > max_y / 2]
        if bottom_amounts:
            return max(bottom_amounts, key=lambda x: x['amount'])['amount']

        # Final fallback: overall largest amount
        return max(all_amounts, key=lambda x: x['amount'])['amount']

    def extract_vat_with_regex(self, layout_data, full_text, all_amounts):
        """Extract VAT amount using regex patterns"""
        all_tax_amounts = extract_all_tax_amounts(layout_data)
        if not all_tax_amounts:
            return None

        # Look for amounts near VAT keywords
        vat_keywords = ["VAT", "TAX", "GST"]

        for i, (text, poly, y_pos) in enumerate(layout_data):
            text_upper = text.upper()

            if any(keyword in text_upper for keyword in vat_keywords):
                # Amount in same line
                line_amounts = [amt for amt in all_tax_amounts if abs(amt['y_pos'] - y_pos) < 5]
                if line_amounts:
                    return max(line_amounts, key=lambda x: x['amount'])['amount']

                # Amount in next line
                if i + 1 < len(layout_data):
                    next_text, next_poly, next_y = layout_data[i + 1]
                    next_amounts = [amt for amt in all_tax_amounts if abs(amt['y_pos'] - next_y) < 5]
                    if next_amounts:
                        return max(next_amounts, key=lambda x: x['amount'])['amount']

        return None
