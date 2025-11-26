import re
from datetime import datetime

def extract_dates_from_text(full_text):
    """Extract dates from receipt text"""
    date_patterns = [
        r"(\d{4}[-/.]\d{2}[-/.]\d{2})",
        r"(\d{2}[-/.]\d{2}[-/.]\d{4})", 
        r"(\d{2}\s*[A-Za-z]{3,9}\s*\d{4})",
        r"([A-Za-z]{3,9}\s*\d{2},?\s*\d{4})",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        for match in matches:
            cleaned_date = clean_and_validate_date(match)
            if cleaned_date:
                return cleaned_date
    return None

def extract_pin_from_text(full_text):
    """Extract PIN numbers from receipt text"""
    pin_patterns = [
        r"(?:PIN\s*[:\-]?\s*([A-Z0-9]{8,15}))",
        r"(?:PIN/VAT\s*[:\-]?\s*([A-Z0-9]{8,15}))",
        r"(?:PIN\s*NO\s*[:\-]?\s*([A-Z0-9]{8,15}))",
        r"(?<!\w)([A-Z]\d{8}[A-Z])(?!\w)",
        r"(?<!\w)(P\d{9}[A-Z])(?!\w)",
    ]

    for pattern in pin_patterns:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        for match in matches:
            cleaned_pin = clean_and_validate_pin(match)
            if cleaned_pin:
                return cleaned_pin
    return None

def clean_and_validate_date(date_string):
    """Clean and validate date strings"""
    if not date_string:
        return None

    cleaned = date_string.strip()
    manual_formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", 
        "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
        "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
    ]

    for fmt in manual_formats:
        try:
            parsed_date = datetime.strptime(cleaned, fmt)
            if 2020 <= parsed_date.year <= 2030:
                return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def clean_and_validate_pin(pin_string):
    """Clean and validate PIN strings"""
    if not pin_string:
        return None

    cleaned = pin_string.strip().upper()
    cleaned = re.sub(r'[-\s.]', '', cleaned)

    if len(cleaned) < 8 or len(cleaned) > 15:
        return None

    pin_patterns = [
        r'^[A-Z]\d{8,10}[A-Z]?$',
        r'^P\d{9,10}[A-Z]?$',
        r'^\d{8,11}$',
    ]

    for pattern in pin_patterns:
        if re.match(pattern, cleaned):
            return cleaned
    return None
