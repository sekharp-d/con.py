import os
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import cv2
import numpy as np
import os
import glob
import pytesseract
import os
import glob
import re
import json
from PIL import Image
from pytesseract import Output
from itertools import groupby
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
from concurrent.futures import ThreadPoolExecutor
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
poppler_path = '/usr/bin/'

height_threshold = 1000

def convert_pdf_to_images(pdf_path, poppler_path):
    with open(pdf_path, "rb") as file:
        pdf = PdfReader(file)
        total_pages = len(pdf.pages)

    images = []
    if total_pages > 3:  # Ensure there are more than 3 pages in the PDF
        images = convert_from_path(pdf_path, first_page=3, last_page=total_pages-1, fmt="jpeg", poppler_path=poppler_path)
    return images


def find_boundaries(lines, length, min_gap, max_count):
    if not lines:
        return []  # Return an empty list if there are no lines

    lines = sorted(lines)
    gaps = [(lines[i], lines[i+1]) for i in range(len(lines) - 1)]
    gaps.append((lines[-1], length))  # Add the last gap
    boundaries = [gap for gap in gaps if gap[1] - gap[0] > min_gap]
    boundaries = sorted(boundaries, key=lambda x: x[1] - x[0], reverse=True)[:max_count]
    boundaries = sorted([boundary[0] for boundary in boundaries])
    return boundaries


def process_image(image, max_columns, max_rows):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    _, binary_image = cv2.threshold(inverted_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Detect vertical lines (columns)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    vertical_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=100, maxLineGap=50)
    filtered_vertical_lines = [line[0][0] for line in vertical_lines if abs(line[0][3] - line[0][1]) > image.shape[0] * 0.5] if vertical_lines is not None else []
    column_boundaries = find_boundaries(filtered_vertical_lines, image.shape[1], image.shape[0] * 0.1, max_columns)

    # Detect horizontal lines (rows)
    horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=100, maxLineGap=50)
    filtered_horizontal_lines = [line[0][1] for line in horizontal_lines if abs(line[0][2] - line[0][0]) > image.shape[1] * 0.5] if horizontal_lines is not None else []
    row_boundaries = find_boundaries(filtered_horizontal_lines, image.shape[0], image.shape[1] * 0.1, max_rows)

    cropped_images = []

    for i, row_start_y in enumerate(row_boundaries[:-1]):  # Skip the last boundary
        next_row_y = row_boundaries[i+1] if i < len(row_boundaries) - 1 else image.shape[0]
        for j, column_start_x in enumerate(column_boundaries):
            next_column_x = column_boundaries[j+1] if j < len(column_boundaries) - 1 else image.shape[1]
            if next_row_y <= row_start_y or next_column_x <= column_start_x:
                print(f"Invalid slice for cell [{i}, {j}]: row {row_start_y}-{next_row_y}, col {column_start_x}-{next_column_x}")
                continue
            cell_image = image[row_start_y:next_row_y, column_start_x:next_column_x]
            if cell_image.size == 0:
                print(f"Empty cell image for cell [{i}, {j}]")
                continue
            cropped_images.append(cell_image)
    return cropped_images


def get_image_height(image_path):
    with Image.open(image_path) as img:
        return img.height


def extract_multiple_person_information(ocr_text):
    identifier_pattern = r'\b[A-Z]{2,}\d{6,}\b'
    identifiers = re.findall(identifier_pattern, ocr_text)

    person_data_array = []
    for identifier in identifiers:
        start = ocr_text.find(identifier)
        end = ocr_text.find(identifiers[identifiers.index(identifier) + 1]) if identifiers.index(identifier) + 1 < len(identifiers) else None
        person_text = ocr_text[start:end].strip()
        if person_text:
            person_data_array.append(extract_information(person_text))
    return person_data_array


def extract_information(ocr_text):
    def extract_field(text, regex):
        match = re.search(regex, text)
        return match.group(1).strip().split("\n")[0].strip() if match else None

    identifier = extract_field(ocr_text, r"([A-Z]{2,}\d{6,})")
    name = extract_field(ocr_text, r"(?:Name|Mame)\s*:\s*([\w\s]+)")
    husband_name = extract_field(ocr_text, r"Husband's\s+(?:Name|Mame)\s*:\s*([\w\s]+)")
    father_name = extract_field(ocr_text, r"Father's\s+(?:Name|Mame)\s*:\s*([\w\s]+)")
    house_number = extract_field(ocr_text, r"House Number\s*:\s*([\w\s-]+)")
    age = extract_field(ocr_text, r"Age\s*:\s*(\d+)")
    gender = extract_field(ocr_text, r"Gender\s*:\s*([\w]+)")

    extracted_data = {
        "Identifier": identifier,
        "Name": name,
        "Husband's Name": husband_name,
        "Father's Name": father_name,
        "House Number": house_number,
        "Age": int(age) if age else None,
        "Gender": gender
    }
    return extracted_data


def perform_ocr(image_path):
    image_height = get_image_height(image_path)
    is_multiple = image_height > height_threshold

    text = pytesseract.image_to_string(image_path, lang='eng', config='--psm 6')
    extracted_data = extract_multiple_person_information(text) if is_multiple else [extract_information(text)]
    print(extracted_data, 'ed')
    return extracted_data


def process_uploaded_pdf(pdf_file_name):
  output = []
  images = convert_pdf_to_images(pdf_file_name, poppler_path)
  print(len(images), 'img')

  def process_image_and_ocr(args):
      i, image = args
      image_path = f"image{i}.jpg"
      image.save(image_path)
      cropped_images = process_image(cv2.imread(image_path), 3, 10)
      print(len(cropped_images), 'cropped img')
      local_output = []
      for j, cropped_image in enumerate(cropped_images):
          cv2.imwrite(f"image{i}_{j}.jpg", cropped_image)
          cropped_image_path = f"image{i}_{j}.jpg"
          extracted_data = perform_ocr(cropped_image_path)
          local_output.extend(extracted_data)
          os.remove(cropped_image_path)
      os.remove(image_path)
      print(local_output, 'lo')
      return local_output

  with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
      results = list(executor.map(process_image_and_ocr, enumerate(images)))
  for result in results:
      output.extend(result)
  os.remove(pdf_file_name)
  return output


# Process the uploaded PDF file
output_data = process_uploaded_pdf(pdf_file_name)

print(output_data, 'od')

# Save the output_data to a JSON file in Google Drive
output_file_path = f'{pdf_file_name}_output.json'
with open(output_file_path, 'w') as f:
    json.dump(output_data, f)

print(f'Output data saved to Google Drive: {output_file_path}')
