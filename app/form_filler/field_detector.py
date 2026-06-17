import re
import fitz
import pdfplumber

def fallback_opencv_detect(doc):
    import cv2
    import numpy as np
    from PIL import Image
    import io
    detected = []
    field_counter = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('RGB')
        img_cv = np.array(img)
        img_cv = img_cv[:, :, ::-1].copy() # RGB to BGR
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        kernel_length = max(10, img_cv.shape[1] // 40)
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        img_temp1 = cv2.erode(thresh, hori_kernel, iterations=1)
        horizontal_lines_img = cv2.dilate(img_temp1, hori_kernel, iterations=1)
        
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        img_temp2 = cv2.erode(thresh, vert_kernel, iterations=1)
        vertical_lines_img = cv2.dilate(img_temp2, vert_kernel, iterations=1)

        alpha = 0.5
        beta = 1.0 - alpha
        img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, np.ones((2,2), np.uint8), iterations=2)
        _, img_final_bin = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Horizontal lines
        contours, _ = cv2.findContours(horizontal_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h < 15:
                fx0, fy0, fx1, fy1 = x/2.0, y/2.0, (x+w)/2.0, (y+h)/2.0
                rect = [fx0, fy0, fx1, fy1]
                
                # Check overlap with existing detected lines
                overlaps = False
                for d in detected:
                    if fitz.Rect(d["rect"]).intersects(fitz.Rect(fx0, fy0 - 5, fx1, fy1 + 5)):
                        overlaps = True
                        break
                if overlaps: continue
                
                context = get_nearest_label(page, rect)
                if not context:
                    # OCR left
                    context = extract_visual_text(page, [max(0, fx0 - 150), max(0, fy0 - 10), fx0, fy1 + 10])
                if not context:
                    context = extract_visual_text(page, [fx0, max(0, fy0 - 25), fx1, fy0])
                if not context:
                    context = ""
                    
                detected.append({
                    "id": f"cv_field_{field_counter}",
                    "type": "visual",
                    "page_index": page_idx,
                    "rect": rect,
                    "context": context
                })
                field_counter += 1
                
        # Boxes
        contours_box, _ = cv2.findContours(~img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_box:
            x, y, w, h = cv2.boundingRect(c)
            if 20 < w < img_cv.shape[1] * 0.9 and h > 20:
                fx0, fy0, fx1, fy1 = x/2.0, y/2.0, (x+w)/2.0, (y+h)/2.0
                overlaps = False
                for d in detected:
                    if fitz.Rect(d["rect"]).intersects(fitz.Rect(fx0, fy0, fx1, fy1)):
                        overlaps = True
                        break
                if not overlaps:
                    rect = [fx0, fy0, fx1, fy1]
                    context = get_nearest_label(page, rect)
                    if not context:
                        context = extract_visual_text(page, [max(0, fx0 - 150), max(0, fy0 - 10), fx0, fy1 + 10])
                    if not context:
                        context = extract_visual_text(page, [fx0, max(0, fy0 - 25), fx1, fy0])
                    if not context:
                        context = ""
                        
                    detected.append({
                        "id": f"cv_field_{field_counter}",
                        "type": "visual",
                        "page_index": page_idx,
                        "rect": rect,
                        "context": context
                    })
                    field_counter += 1
                    
    return detected


def extract_visual_text(fitz_page, rect):
    try:
        from PIL import Image
        import io
        
        x0, y0, x1, y1 = rect
        clip_rect = fitz.Rect(max(0, x0-2), max(0, y0-2), x1+2, y1+2)
        mat = fitz.Matrix(3.0, 3.0)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip_rect)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        try:
            import pytesseract
            text = pytesseract.image_to_string(img, lang='jpn+eng')
            return text.strip()
        except Exception:
            # Fallback to winocr for local Windows debugging
            import winocr
            import concurrent.futures
            
            # Detect language from document path/name
            doc_path = ""
            try:
                if hasattr(fitz_page, "parent") and fitz_page.parent:
                    doc_path = getattr(fitz_page.parent, "name", "").lower()
            except Exception:
                pass
                
            lang = 'en'
            if any(k in doc_path for k in ['nihongo', 'jpn', 'japanese', 'n3', 'moji', 'goi']) or re.search(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf]', doc_path):
                lang = 'ja'
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                res = executor.submit(winocr.recognize_pil_sync, img, lang).result()
            return res.get("text", "").strip()
    except Exception as e:
        print("Visual OCR failed:", e)
        return ""

def get_closest_phrase(word_list, target_rect, direction='left'):
    if not word_list:
        return None, float("inf"), None
        
    tx0, ty0, tx1, ty1 = target_rect
    
    # Group by top coordinate (within 3 pt)
    word_list.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    curr_line = []
    for w in word_list:
        if not curr_line:
            curr_line.append(w)
        else:
            if abs(w["top"] - curr_line[-1]["top"]) < 3:
                curr_line.append(w)
            else:
                lines.append(curr_line)
                curr_line = [w]
    if curr_line:
        lines.append(curr_line)
        
    valid_lines = []
    for line in lines:
        phrase_text = " ".join([w["text"] for w in line])
        # Skip if only punctuation or empty
        if re.match(r"^[\s_:\-\.>=<]+$", phrase_text):
            continue
        valid_lines.append(line)
        
    if not valid_lines:
        return None, float("inf"), None
        
    best_idx = -1
    min_d = float("inf")
    
    for i, line in enumerate(valid_lines):
        if direction == 'left':
            phrase_x1 = max(w["x1"] for w in line)
            d = tx0 - phrase_x1
        elif direction == 'right':
            phrase_x0 = min(w["x0"] for w in line)
            d = phrase_x0 - tx1
        else:
            phrase_bottom = max(w["bottom"] for w in line)
            d = ty0 - phrase_bottom
            
        if d < min_d:
            min_d = d
            best_idx = i
            
    if best_idx == -1:
        return None, float("inf"), None
        
    # We purposefully DO NOT expand to capture multi-line paragraphs
    # to avoid merging huge header blocks into a single field label.
    merged_lines = [valid_lines[best_idx]]
                
    best_phrase = "\n".join([" ".join([w["text"] for w in line]) for line in merged_lines])
    min_x0 = min(min(w["x0"] for w in line) for line in merged_lines)
    min_top = min(min(w["top"] for w in line) for line in merged_lines)
    max_x1 = max(max(w["x1"] for w in line) for line in merged_lines)
    max_bottom = max(max(w["bottom"] for w in line) for line in merged_lines)
    best_box = (min_x0, min_top, max_x1, max_bottom)
    
    return best_phrase, min_d, best_box

def ocr_get_all_words(fitz_page):
    try:
        mat = fitz.Matrix(2.0, 2.0)
        pix = fitz_page.get_pixmap(matrix=mat)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('RGB')
        
        try:
            import pytesseract
            data = pytesseract.image_to_data(img, lang='nep+jpn+eng', output_type=pytesseract.Output.DICT)
            ocr_words = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue
                wx0 = data['left'][i] / 2.0
                wtop = data['top'][i] / 2.0
                wx1 = (data['left'][i] + data['width'][i]) / 2.0
                wbottom = (data['top'][i] + data['height'][i]) / 2.0
                ocr_words.append((wx0, wtop, wx1, wbottom, text, 0, 0, 0))
            return ocr_words
        except Exception:
            import winocr
            import concurrent.futures
            
            doc_path = ""
            try:
                if hasattr(fitz_page, "parent") and fitz_page.parent:
                    import os
                    doc_path = os.path.basename(getattr(fitz_page.parent, "name", "")).lower()
            except Exception:
                pass
                
            lang = 'en'
            if any(k in doc_path for k in ['nihongo', 'jpn', 'japanese', 'n3', 'moji', 'goi']) or re.search(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf]', doc_path):
                lang = 'ja'
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                res = executor.submit(winocr.recognize_pil_sync, img, lang).result()
                
            ocr_words = []
            for line in res.get("lines", []):
                for word in line.get("words", []):
                    text = word.get("text", "").strip()
                    if not text:
                        continue
                    rect = word.get("bounding_rect", {})
                    wx0 = rect.get("x", 0.0) / 2.0
                    wtop = rect.get("y", 0.0) / 2.0
                    wx1 = (rect.get("x", 0.0) + rect.get("width", 0.0)) / 2.0
                    wbottom = (rect.get("y", 0.0) + rect.get("height", 0.0)) / 2.0
                    ocr_words.append((wx0, wtop, wx1, wbottom, text, 0, 0, 0))
            return ocr_words
    except Exception as e:
        print("ocr_get_all_words failed:", e)
        return []

def get_nearest_label(page, target_rect, max_dist_x=200, max_dist_y=60):
    """
    Finds the nearest text block to the left or above the target_rect.
    Returns the clean label text or an empty string.
    """
    import re
    _garbage_chars = set('|{}[]\\^+={}<>@#$%&*~`;/')
    _vowels = set('aeiouAEIOU')
    _valid_no_vowels = {'by', 'By', 'BY', 'Mr', 'Mrs', 'Dr', 'St', 'No', 'Id', 'ID', 'Ref', 'ref', 'Qty'}

    def _is_english_token(tok):
        if not tok:
            return False
        specials = sum(1 for c in tok if c in _garbage_chars)
        if specials > 0:
            return False
        letters = sum(1 for c in tok if c.isalpha())
        if letters == 0:
            return False
        # All digits or digits with punctuation are ok (e.g. account numbers, dates)
        if all(c.isdigit() or c in '.,-/' for c in tok):
            return True
        # Preeti garbage is mostly consonant clusters with no vowels
        has_vowel = any(c in _vowels for c in tok)
        if not has_vowel and tok not in _valid_no_vowels and len(tok) > 2:
            return False
        return True

    def clean_japanese_garbage(text):
        if not text:
            return text
        # Remove non-ASCII non-Devanagari unicode
        text = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', '', text).strip()
        # Filter Preeti-encoded garbage (legacy Nepali font ASCII chars like k|fylds, vftf, g+=)
        tokens = text.split()
        english_tokens = [tok for tok in tokens if _is_english_token(tok)]
        result = ' '.join(english_tokens).strip()
        return result if result else text
        
    tx0, ty0, tx1, ty1 = target_rect
    t_cy = (ty0 + ty1) / 2
    
    words = page.get_text("words")
    if not words:
        if not hasattr(page, "_cached_ocr_words"):
            page._cached_ocr_words = ocr_get_all_words(page)
        words = page._cached_ocr_words
    left_words = []
    right_words = []
    above_words = []
    
    for w in words:
        wx0, wtop, wx1, wbottom, text, block_no, line_no, word_no = w
        w_cy = (wtop + wbottom) / 2
        
        # For underlines, text often sits on top of the line, so w_cy can be much higher than ty0
        # Check left
        if wx1 <= tx0 + 10 and ty0 - 25 <= w_cy <= ty1 + 15:
            left_words.append({"x0": wx0, "top": wtop, "x1": wx1, "bottom": wbottom, "text": text})
        # Check right
        elif wx0 >= tx1 - 10 and ty0 - 25 <= w_cy <= ty1 + 15:
            right_words.append({"x0": wx0, "top": wtop, "x1": wx1, "bottom": wbottom, "text": text})
        # Check above
        elif wbottom <= ty0 + 10 and ty0 - wbottom < max_dist_y:
            if wx0 <= tx1 + 20 and wx1 >= tx0 - 20:
                above_words.append({"x0": wx0, "top": wtop, "x1": wx1, "bottom": wbottom, "text": text})
                
    left_phrase, left_dist, left_box = get_closest_phrase(left_words, target_rect, direction='left')
    right_phrase, right_dist, right_box = get_closest_phrase(right_words, target_rect, direction='right')
    above_phrase, above_dist, above_box = get_closest_phrase(above_words, target_rect, direction='above')
    
    best_label = ""
    best_rect = None
    
    if left_phrase and left_dist < max_dist_x:
        best_label = left_phrase
        best_rect = left_box
    elif right_phrase and right_dist < max_dist_x:
        best_label = right_phrase
        best_rect = right_box
    elif above_phrase:
        best_label = above_phrase
        best_rect = above_box
    else:
        best_label = left_phrase or right_phrase or ""
        best_rect = left_box or right_box

    if best_label and best_rect:
        ocr_text = extract_visual_text(page, best_rect)
        if ocr_text:
            best_label = ocr_text

    if best_label:
        best_label = re.sub(r"__+", "", best_label)
        best_label = best_label.strip().rstrip(":").rstrip(" ").rstrip("-").strip()
        best_label = re.sub(r"\s+", " ", best_label)
        
        # Suppress generic labels
        generic_words = {
            "blank line", "box", "table grid", "visual", "field", "text", 
            "textfield", "formfield", "textbox", "input", "value", "button",
            "select", "checkbox", "radio", "underline"
        }
        if best_label.lower() in generic_words:
            best_label = ""
        
    return clean_japanese_garbage(best_label)

def get_table_label(pdfplumber_page, table_bbox, fitz_page=None):
    """
    Finds the nearest text label for a table by identifying the closest
    horizontal/vertical phrase using distance minimization.
    """
    tx0, t_top, tx1, t_bottom = table_bbox
    words = pdfplumber_page.extract_words()
    
    left_words = []
    above_words = []
    
    for w in words:
        wx0, wtop, wx1, wbottom = w["x0"], w["top"], w["x1"], w["bottom"]
        w_cy = (wtop + wbottom) / 2
        
        # Check left
        if wx1 <= tx0 + 5 and t_top - 10 <= w_cy <= t_bottom + 10:
            left_words.append(w)
        # Check above
        elif wbottom <= t_top + 5 and t_top - wbottom < 40:
            if wx0 <= tx1 + 10 and wx1 >= tx0 - 10:
                above_words.append(w)
                
    left_phrase, left_dist, left_box = get_closest_phrase(left_words, table_bbox, direction='left')
    above_phrase, above_dist, above_box = get_closest_phrase(above_words, table_bbox, direction='above')
    
    # Prefer left phrase if it is within 80 points
    label_box = None
    if left_phrase and left_dist < 80:
        label = left_phrase
        label_box = left_box
    elif above_phrase:
        label = above_phrase
        label_box = above_box
    else:
        label = left_phrase or ""
        label_box = left_box
        

    if label:
        import re
        _garbage_chars = set('|{}[]\\^+={}<>@#$%&*~`;/')
        _vowels = set('aeiouAEIOU')
        _valid_no_vowels = {'by', 'By', 'BY', 'Mr', 'Mrs', 'Dr', 'St', 'No', 'Id', 'ID', 'Ref', 'ref', 'Qty'}
        label = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', '', label)
        tokens = label.split()
        english_tokens = []
        for tok in tokens:
            if not tok:
                continue
            specials = sum(1 for c in tok if c in _garbage_chars)
            if specials > 0:
                continue
            letters = sum(1 for c in tok if c.isalpha())
            if letters == 0:
                continue
            has_vowel = any(c in _vowels for c in tok)
            if not has_vowel and tok not in _valid_no_vowels and len(tok) > 2:
                continue
            english_tokens.append(tok)
        if english_tokens:
            label = ' '.join(english_tokens)
        label = re.sub(r"__+", "", label)
        label = label.strip().rstrip(":").rstrip(" ").rstrip("-").strip()
            
    return label or "Table Grid"

def detect_fields(pdf_path: str):
    """
    Parses PDF to detect interactive AcroForm fields, visual character-grid tables,
    and visual blanks (underlines/boxes).
    """
    doc = fitz.open(pdf_path)
    detected = []
    field_counter = 0
    
    # Track table regions per page to avoid duplicate field detection in those areas
    table_rects_by_page = {}
    
    # 1. Detect Character-Grid Tables using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, pl_page in enumerate(pdf.pages):
            table_rects_by_page[page_idx] = []
            tables = pl_page.find_tables()
            
            for table in tables:
                tx0, t_top, tx1, t_bottom = table.bbox
                width = tx1 - tx0
                height = t_bottom - t_top
                
                if height < 50 and len(table.cells) >= 3:
                    table_rects_by_page[page_idx].append(fitz.Rect(tx0, t_top, tx1, t_bottom))
                    
                    # Expand bounding box using fitz drawings to recover missed cells
                    # pdfplumber often misses the first/last cells if text is written over the lines
                    drawings = doc[page_idx].get_drawings()
                    min_x, max_x = tx0, tx1
                    for d in drawings:
                        r = d["rect"]
                        # If a drawing line aligns perfectly with the top or bottom of our table
                        if abs(r.y0 - t_top) < 3 or abs(r.y1 - t_bottom) < 3:
                            # And it's roughly connected or adjacent to our table's x bounds
                            if r.x1 >= min_x - 50 and r.x0 <= max_x + 50:
                                min_x = min(min_x, r.x0)
                                max_x = max(max_x, r.x1)
                    
                    tx0 = min_x
                    tx1 = max_x
                    width = tx1 - tx0

                    # Generate perfectly uniform cells mathematically to fix pdfplumber merging errors
                    sorted_cells = sorted(table.cells, key=lambda c: c[0])
                    widths = [c[2] - c[0] for c in sorted_cells]
                    
                    import statistics
                    median_w = statistics.median(widths) if widths else 15.0
                    if median_w < 5.0: median_w = 15.0
                    
                    # Estimate the number of uniform blocks that should fit in this table
                    num_expected_cells = max(1, round(width / median_w))
                    cell_w = width / num_expected_cells
                    
                    cells_list = []
                    for i in range(num_expected_cells):
                        cells_list.append([
                            tx0 + (i * cell_w),
                            t_top,
                            tx0 + ((i + 1) * cell_w),
                            t_bottom
                        ])
                        
                    # Find context label
                    context = get_table_label(pl_page, (tx0, t_top, tx1, t_bottom), fitz_page=doc[page_idx])
                    
                    detected.append({
                        "id": f"field_{field_counter}",
                        "type": "character-grid",
                        "page_index": page_idx,
                        "rect": [tx0, t_top, tx1, t_bottom],
                        "cells": cells_list,
                        "context": context
                    })
                    field_counter += 1

    # Process page by page with fitz for AcroForm and other visual fields
    for page_idx, page in enumerate(doc):
        page_table_rects = table_rects_by_page.get(page_idx, [])
        widget_rects = []
        
        # 2. Detect AcroForm Widgets
        widgets = page.widgets()
        if widgets:
            for widget in widgets:
                rect = [widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1]
                
                # Check if it overlaps with a character grid table
                in_table = False
                for t_r in page_table_rects:
                    if widget.rect.intersects(t_r):
                        in_table = True
                        break
                if in_table:
                    continue
                    
                widget_rects.append(widget.rect)
                
                context = get_nearest_label(page, rect)
                if not context:
                    context = widget.field_name or f"Field {field_counter}"
                    
                detected.append({
                    "id": f"field_{field_counter}",
                    "type": "digital",
                    "page_index": page_idx,
                    "rect": rect,
                    "context": context,
                    "field_name": widget.field_name,
                    "field_type": widget.field_type
                })
                field_counter += 1
                
        # 3. Detect Underscores (___) and Dotted Lines (.....)
        underscore_rects = []
        words = page.get_text("words")
        current_rect = None
        
        # Filter words that are purely dots or underscores
        dot_words = []
        for w in words:
            text = w[4]
            if re.match(r"^[\._\-]{2,}$", text):
                dot_words.append(fitz.Rect(w[:4]))
                
        # Also keep the search_for just in case of weird spacing
        matches = page.search_for("___") + page.search_for("...") + page.search_for("....") + page.search_for(".....") + page.search_for(". . .")
        for r in matches:
            dot_words.append(fitz.Rect(r))
            
        if dot_words:
            dot_words.sort(key=lambda r: (r.y0, r.x0))
            
            for r in dot_words:
                # Filter out if intersects table grid or widget
                overlaps = False
                expanded_r = fitz.Rect(r.x0, r.y0 - 2, r.x1, r.y1 + 2)
                for t_r in page_table_rects:
                    if expanded_r.intersects(t_r):
                        overlaps = True
                        break
                for w_r in widget_rects:
                    if expanded_r.intersects(w_r):
                        overlaps = True
                        break
                if overlaps:
                    continue
                    
                if current_rect is None:
                    current_rect = fitz.Rect(r)
                else:
                    # Merge horizontal runs
                    if abs(r.y0 - current_rect.y0) < 5 and r.x0 <= current_rect.x1 + 15:
                        current_rect.x1 = max(current_rect.x1, r.x1)
                        current_rect.y0 = min(current_rect.y0, r.y0)
                        current_rect.y1 = max(current_rect.y1, r.y1)
                    else:
                        underscore_rects.append(current_rect)
                        current_rect = fitz.Rect(r)
            if current_rect is not None:
                underscore_rects.append(current_rect)
                
        for r in underscore_rects:
            rect = [r.x0, r.y0, r.x1, r.y1]
            context = get_nearest_label(page, rect)
            if not context:
                context = ""
            detected.append({
                "id": f"field_{field_counter}",
                "type": "visual",
                "page_index": page_idx,
                "rect": rect,
                "context": context
            })
            field_counter += 1
            
        # 4. Detect Horizontal Drawing Lines
        drawings = page.get_drawings()
        line_rects = []
        for draw in drawings:
            if draw["type"] == "s" or draw["type"] == "fs":
                rect = draw["rect"]
                width = rect.x1 - rect.x0
                height = rect.y1 - rect.y0
                if height < 2 and width >= 30:
                    overlaps = False
                    # Expand the rect slightly to ensure thin lines lying exactly on borders intersect properly
                    expanded_rect = fitz.Rect(rect.x0, rect.y0 - 2, rect.x1, rect.y1 + 2)
                    for t_r in page_table_rects:
                        if expanded_rect.intersects(t_r):
                            overlaps = True
                            break
                    for w_r in widget_rects:
                        if expanded_rect.intersects(w_r):
                            overlaps = True
                            break
                    for u_r in underscore_rects:
                        if expanded_rect.intersects(u_r):
                            overlaps = True
                            break
                    if not overlaps:
                        line_rects.append(rect)
                        
        for r in line_rects:
            rect = [r.x0, r.y0, r.x1, r.y1]
            context = get_nearest_label(page, rect)
            if not context:
                context = ""
            detected.append({
                "id": f"field_{field_counter}",
                "type": "visual",
                "page_index": page_idx,
                "rect": rect,
                "context": context
            })
            field_counter += 1
            
        # 5. Detect Empty Drawing Rectangles (Boxes)
        box_rects = []
        for draw in drawings:
            if draw["type"] == "s" or draw["type"] == "fs":
                rect = draw["rect"]
                width = rect.x1 - rect.x0
                height = rect.y1 - rect.y0
                if width >= 30 and 10 <= height <= 40:
                    text_inside = page.get_text("text", clip=rect).strip()
                    if not text_inside:
                        overlaps = False
                        for t_r in page_table_rects:
                            if rect.intersects(t_r):
                                overlaps = True
                                break
                        for w_r in widget_rects:
                            if rect.intersects(w_r):
                                overlaps = True
                                break
                        for u_r in underscore_rects:
                            if rect.intersects(u_r):
                                overlaps = True
                                break
                        for l_r in line_rects:
                            if rect.intersects(l_r):
                                overlaps = True
                                break
                        if not overlaps:
                            box_rects.append(rect)
                            
        for r in box_rects:
            rect = [r.x0, r.y0, r.x1, r.y1]
            context = get_nearest_label(page, rect)
            if not context:
                context = ""
            detected.append({
                "id": f"field_{field_counter}",
                "type": "visual",
                "page_index": page_idx,
                "rect": rect,
                "context": context
            })
            field_counter += 1
            
    if not detected:
        try:
            detected = fallback_opencv_detect(doc)
        except Exception as e:
            print(f"OpenCV fallback detection failed: {e}")
            
    doc.close()
    return detected
