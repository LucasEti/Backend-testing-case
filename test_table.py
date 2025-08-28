import pytest
from Detector import TableDetector
import os

# --- Initialize the detector once for all tests ---
detector = TableDetector()

# --- Paths setup ---
BASE_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, "images")

def test_success_invoice():
    """
    Should detect at least one table in an invoice image.
    - Loads 'invoice.png' from the images folder
    """
    image_path = os.path.join(IMG_DIR, "invoice.png")
    result = detector.predict(image_path)
    assert isinstance(result, list)
    assert len(result) > 0
    assert "box" in result[0]

def test_success_bank_document():
    """
    Should detect a table in a bank document
    - Loads : 'bank.png'
    """
    image_path = os.path.join(IMG_DIR, "bank.png")
    result = detector.predict(image_path)
    assert len(result) > 0

def test_no_table():
    """
    Test the extraction error with an image wich is not an invoice or bank document
    Image with no table should raise RuntimeError
    """
    image_path = os.path.join(IMG_DIR, "noTable.png")
    with pytest.raises(RuntimeError):
        detector.predict(image_path)

def test_invalid_path():
    """Invalid path test
    - Calls detector.predict() with a non-existing file
    """
    with pytest.raises(RuntimeError):
        detector.predict("invalid_path.jpg")