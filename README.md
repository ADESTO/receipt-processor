# Receipt Processor with VAT Extraction

A web-based tool that extracts total amount, VAT, date, and PIN from receipt images using OCR and machine learning.

## Features

- 📷 OCR text extraction from receipt images
- 💰 Total amount detection with ML
- 🧾 VAT amount extraction
- 📅 Date extraction
- 🔑 PIN number extraction
- 🌐 Web interface via Gradio

## How to Use

1. Upload a receipt image (JPG, PNG, JPEG)
2. Click "Process Receipt"
3. View extracted information:
   - Total Amount
   - VAT Amount
   - Transaction Date
   - Business PIN

## Live Demo

Try it on [Hugging Face Spaces](your-hf-space-link)

## Technical Details

- **OCR**: PaddleOCR for text detection
- **ML**: Random Forest classifiers for amount detection
- **Web Framework**: Gradio for easy deployment
- **Processing**: Regex patterns + ML confidence scoring

## Supported Receipt Types

- Retail receipts
- Restaurant bills
- Supermarket receipts
- Various formats with total amounts
