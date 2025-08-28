# Table Detector (DETR)

This project provides a table detection system based on DETR. 
It uses a fine-tuned model [`TahaDouaji/detr-doc-table-detection`](https://huggingface.co/TahaDouaji/detr-doc-table-detection) from HuggingFace to detect tables in scanned documents (invoices, bank statements).

# Features 

- Detect Tables in images
- Confidence score (threshold)
- Unit tests with PyTest

# Installation

- Go in the root of the repertory and run in a terminal "pip install -r requirements.txt"
- You must have Python

# Run the tests with 

- If you just want to run the test, use "pytest"
- If you want to see the logs, use "pytest -s"
- If you want to see the details, use "pytest -v"
- If you want to generate an HTML report, use : 
  - "pytest --html=reports/report_$(date +%Y-%m-%d_%H-%M-%S).html --self-contained-html" on MAC
  - "pytest --html=reports/report_$(Get-Date -Format "yyyy-MM-dd_HH-mm-ss").html --self-contained-html" on Windows





