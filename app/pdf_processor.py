import fitz  # PyMuPDF

def get_pdf_page_count(file_path: str) -> int:
    """Gets the total page count of a PDF file."""
    with fitz.open(file_path) as doc:
        return len(doc)

def render_pdf_page(file_path: str, page_index: int, thumbnail: bool = False) -> bytes:
    """Renders a specific page of a PDF file to PNG bytes."""
    with fitz.open(file_path) as doc:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"Page index {page_index} out of range (0-{len(doc)-1})")
        page = doc[page_index]
        if thumbnail:
            matrix = fitz.Matrix(0.4, 0.4)
        else:
            matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")
