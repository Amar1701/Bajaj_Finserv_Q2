from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import easyocr
import numpy as np
import cv2
import io
import re

app = FastAPI()
reader = easyocr.Reader(['en'], gpu=False)


def preprocess_image(image_pil):
    image = np.array(image_pil)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def extract_lab_tests_with_refs(text_lines):
    results = []
    ref_ranges = {}

    # Step 1: Separate lines containing reference ranges
    for line in text_lines:
        line = line.strip()
        # Pattern like: 4.7 - 22.1 or 0.00 - 3.55
        match = re.match(r'([0-9.]+)\s*[-–]\s*([0-9.]+)', line)
        if match:
            ref_ranges[line] = (float(match.group(1)), float(match.group(2)))

    # Step 2: Extract test name, value, unit from other lines
    for line in text_lines:
        line = line.strip()
        if len(line) < 5 or re.match(r'^\d{5,}$', line):
            continue

        # Match: TestName Value Unit
        match = re.match(r'([A-Z /()]+)\s+([\d.]+)\s*([a-zA-Z/%]*)$', line)
        if match:
            test_name = match.group(1).strip()
            test_value = float(match.group(2))
            unit = match.group(3).strip() if match.group(3) else ""

            # Try to find closest reference range (same line or nearby)
            ref_low, ref_high, ref_str = None, None, ""
            for ref_line, (low, high) in ref_ranges.items():
                if abs(test_value - low) < 10 or abs(test_value - high) < 10:
                    ref_low, ref_high = low, high
                    ref_str = f"{low} - {high}"
                    break

            out_of_range = False
            if ref_low is not None and ref_high is not None:
                out_of_range = not (ref_low <= test_value <= ref_high)

            results.append({
                "test_name": test_name,
                "test_value": str(test_value),
                "bio_reference_range": ref_str,
                "test_unit": unit,
                "lab_test_out_of_range": out_of_range
            })

    return results


@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")

        preprocessed_img = preprocess_image(image)

        # OCR: Extract text lines
        text_lines = reader.readtext(preprocessed_img, detail=0)

        # Extract test + reference pairs
        parsed_data = extract_lab_tests_with_refs(text_lines)

        return JSONResponse(content={
            "is_success": True,
            "data": parsed_data
        })

    except Exception as e:
        return JSONResponse(content={
            "is_success": False,
            "error": str(e)
        })
