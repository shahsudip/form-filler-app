import io
import os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter
from app.form_filler.field_detector import detect_fields

# Register fonts globally
has_japanese = False
has_nepali = False

jap_fonts = ["C:/Windows/Fonts/msgothic.ttc", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"]
for path in jap_fonts:
    if os.path.exists(path):
        if path.endswith(".ttc"):
            pdfmetrics.registerFont(TTFont("JapaneseFont", path, subfontIndex=0))
        else:
            pdfmetrics.registerFont(TTFont("JapaneseFont", path))
        has_japanese = True
        break

nep_fonts = ["C:/Windows/Fonts/Nirmala.ttf", "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf", "/usr/share/fonts/truetype/fonts-deva.ttf"]
for path in nep_fonts:
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont("NepaliFont", path))
        has_nepali = True
        break

def get_font_for_char(char: str) -> str:
    """Returns the best font name based on the Unicode point of the character."""
    code = ord(char)
    # Devanagari block
    if 0x0900 <= code <= 0x097F and has_nepali:
        return "NepaliFont"
    # Japanese blocks
    if 0x3040 <= code <= 0x9FAF and has_japanese:
        return "JapaneseFont"
    if 0xFF00 <= code <= 0xFFEF and has_japanese:
        return "JapaneseFont"
    return "Helvetica"

def fill_pdf(input_pdf_path: str, output_pdf_path: str, answers: list, pen_color: str = "#000000"):
    """
    Fills digital AcroForm fields, visual blank lines/boxes, and character-grid tables in the input PDF.
    Renders visual overlays in the specified pen color and saves the output.
    """
    # 1. Detect all fields to get coordinates, types, names, pages, and cell locations
    fields_list = detect_fields(input_pdf_path)
    fields_by_id = {f["id"]: f for f in fields_list}

    # 2. Open PDF with pypdf
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    
    # Copy all pages to writer
    for page in reader.pages:
        writer.add_page(page)

    # Group answers by page index
    visual_answers_by_page = {}  # page_idx -> list of (field_info, value)
    digital_answers_by_page = {}  # page_idx -> dict of {field_name: value}

    for ans in answers:
        field_id = ans.get("field_id")
        value = ans.get("value", "")
        if value is None or value == "":
            continue
            
        field_info = fields_by_id.get(field_id)
        if not field_info:
            continue
            
        page_idx = field_info["page_index"]
        if field_info["type"] == "digital":
            if page_idx not in digital_answers_by_page:
                digital_answers_by_page[page_idx] = {}
            # Use the raw field_name
            digital_answers_by_page[page_idx][field_info["field_name"]] = str(value)
        else:
            if page_idx not in visual_answers_by_page:
                visual_answers_by_page[page_idx] = []
            visual_answers_by_page[page_idx].append((field_info, str(value)))

    # 3. Apply digital answers
    for page_idx, field_values in digital_answers_by_page.items():
        page = writer.pages[page_idx]
        writer.update_page_form_field_values(page, field_values)

    # 4. Generate visual overlays (blanks, lines, boxes, and character grids) page by page
    try:
        color = HexColor(pen_color)
    except Exception:
        color = HexColor("#000000")

    for page_idx, items in visual_answers_by_page.items():
        page = writer.pages[page_idx]
        
        # mediabox width and height represent physical dimensions of page in points
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        
        # Create a transparent overlay canvas
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(width, height))
        can.setFillColor(color)
        
        font_size = 10
        can.setFont("Helvetica", font_size)
        
        for field_info, value in items:
            field_type = field_info["type"]
            
            if field_type == "character-grid":
                cells = field_info.get("cells", [])
                # Split answer into individual characters, forcing English to uppercase
                chars = list(value.upper())
                for i, char in enumerate(chars):
                    if i >= len(cells):
                        break
                    cx0, ctop, cx1, cbottom = cells[i]
                    # Compute center coordinates
                    cx = (cx0 + cx1) / 2.0
                    cy = (ctop + cbottom) / 2.0
                    
                    cell_width = cx1 - cx0
                    cell_height = cbottom - ctop
                    font_size = min(cell_width * 0.65, cell_height * 0.65)
                    
                    cx_rl = cx
                    # Center text vertically by shifting baseline down by 0.35 * font_size
                    cy_rl = height - cy - (font_size * 0.35)
                    
                    font_name = get_font_for_char(char)
                    can.setFont(font_name, font_size)
                    can.drawCentredString(cx_rl, cy_rl, char)
            else:
                # Standard visual blank (underline or box)
                rect = field_info["rect"]
                x0, y0, x1, y1 = rect
                
                box_height = y1 - y0
                is_line = box_height < 10.0
                
                if is_line:
                    font_size = 10.0
                else:
                    font_size = box_height * 0.65
                    if font_size > 14: font_size = 14
                    if font_size < 8: font_size = 8
                
                can.saveState()
                if not is_line:
                    path = can.beginPath()
                    # rect in reportlab is (x, y, width, height) from bottom-left
                    path.rect(x0, height - y1, x1 - x0, y1 - y0)
                    can.clipPath(path, stroke=0, fill=0)

                # Center horizontally
                x_rl = x0 + 2
                
                # Convert y coordinate from top-left (PDF) to bottom-left (ReportLab)
                # Place text baseline slightly above the visual underline (y1)
                y_rl = height - y1 + 2.0
                
                best_font = "Helvetica"
                for ch in value:
                    f = get_font_for_char(ch)
                    if f != "Helvetica":
                        best_font = f
                        break
                        
                can.setFont(best_font, font_size)
                can.drawString(x_rl, y_rl, value)
                can.restoreState()
            
        can.save()
        packet.seek(0)
        
        # Load temporary overlay page
        overlay_reader = PdfReader(packet)
        overlay_page = overlay_reader.pages[0]
        
        # Merge transparent text overlay onto original page
        page.merge_page(overlay_page)

    # Write the completed PDF out
    with open(output_pdf_path, "wb") as f:
        writer.write(f)
