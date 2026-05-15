from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Test", ln=True, align="C")

# Output as string, encode as latin-1
pdf_bytes = pdf.output(dest='S').encode('latin-1')
print(f"Type: {type(pdf_bytes)}, Length: {len(pdf_bytes)}")
