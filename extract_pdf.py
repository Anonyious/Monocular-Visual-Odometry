import pdfplumber

with pdfplumber.open("Monocular_visual_odometry.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}\n")
    for i, page in enumerate(pdf.pages[:15]):  # First 15 pages
        text = page.extract_text()
        print(f"=== PAGE {i+1} ===\n")
        print(text[:2000] if text else "[No text]")
        print("\n" + "="*80 + "\n")
