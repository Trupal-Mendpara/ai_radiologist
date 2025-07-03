from fpdf import FPDF
import base64
from io import BytesIO

def generate_pdf_report(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)

    # Generate PDF as a string and write to BytesIO
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # FPDF returns str; encode to bytes
    pdf_output = BytesIO(pdf_bytes)

    return pdf_output