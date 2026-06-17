import os
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
import io
import asyncio
import json
import shutil
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.schemas import (
    UploadResponse,
    PdfHistoryItem,
    FormFillerUploadResponse,
    FormField,
    FormFillerFillRequest,
    ProfileEntry,
    ProfileResponse,
    ProfileSaveRequest,
    SuggestRequest,
    SuggestResponse,
    SuggestionResult,
)
from app.pdf_processor import get_pdf_page_count, render_pdf_page
from app.form_filler.field_detector import detect_fields
from app.form_filler.question_generator import generate_question
from app.form_filler.pdf_writer import fill_pdf
from app.form_filler.vector_store import ProfileVectorStore

app = FastAPI(
    title="PDF Page Viewer API",
    description="API to upload a PDF file and stream rendered page images.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
HISTORY_FILE = os.path.join(CACHE_DIR, "_history.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Module-level ProfileVectorStore (index rebuilt lazily when profile changes)
_profile_store = ProfileVectorStore(profile_dir=CACHE_DIR)


def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _add_to_history(filename: str, total_pages: int, file_size_bytes: int):
    history = _load_history()
    # Update existing entry if same filename, else prepend new entry
    history = [h for h in history if h.get("filename") != filename]
    history.insert(0, {
        "filename": filename,
        "total_pages": total_pages,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": file_size_bytes,
    })
    _save_history(history)


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.get("/pdf-history", tags=["History"])
async def get_pdf_history():
    """Returns list of previously uploaded PDFs that are still in cache."""
    history = _load_history()
    # Filter to only PDFs that still physically exist in the cache
    valid = [h for h in history if os.path.exists(os.path.join(CACHE_DIR, os.path.basename(h["filename"])))]
    if len(valid) != len(history):
        _save_history(valid)
    return {"history": valid}


@app.post("/select-pdf", response_model=UploadResponse, tags=["History"])
async def select_pdf_from_history(filename: str = Query(..., description="Filename from history to open")):
    """Opens an already-cached PDF from history without re-uploading."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(CACHE_DIR, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found in cache. Please upload it again.")
    total_pages = get_pdf_page_count(file_path)
    return UploadResponse(filename=safe_filename, total_pages=total_pages)


async def _process_upload_to_pdf(file: UploadFile) -> tuple[str, str]:
    filename = os.path.basename(file.filename)
    is_image = filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    
    if not (filename.lower().endswith('.pdf') or is_image):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and Image files are supported.")
        
    if is_image:
        filename = os.path.splitext(filename)[0] + ".pdf"
        
    dest_path = os.path.join(CACHE_DIR, filename)
    content = await file.read()
    
    if is_image:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(content)).convert('RGB')
        img.save(dest_path, "PDF", resolution=100.0)
    else:
        with open(dest_path, "wb") as buffer:
            buffer.write(content)
            
    return dest_path, filename


@app.post("/upload-pdf", response_model=UploadResponse, tags=["PDF"])
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF or Image file, converts if necessary, and returns details."""
    try:
        dest_path, filename = await _process_upload_to_pdf(file)

        total_pages = get_pdf_page_count(dest_path)
        file_size_bytes = os.path.getsize(dest_path)
        _add_to_history(filename, total_pages, file_size_bytes)

        return UploadResponse(filename=filename, total_pages=total_pages)
    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading/processing the PDF: {str(e)}")


@app.get("/page-image", tags=["PDF"])
async def get_page_image(
    filename: str = Query(..., description="Name of the PDF file in cache"),
    page_index: int = Query(..., description="0-based index of the page to render"),
    thumbnail: bool = Query(False, description="Whether to render a low-res thumbnail")
):
    """Loads the PDF from cache, renders the specified page, and streams it as PNG."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(CACHE_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found in cache. Please upload it first.")

    try:
        img_bytes = render_pdf_page(file_path, page_index, thumbnail)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except IndexError as idx_err:
        raise HTTPException(status_code=400, detail=str(idx_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while rendering page: {str(e)}")


@app.get("/page-text", tags=["PDF"])
def get_page_text(
    filename: str = Query(..., description="Name of the PDF file in cache"),
    page_index: int = Query(..., description="0-based index of the page to extract text from"),
    use_ocr: bool = Query(True, description="Whether to use OCR for text extraction")
):
    """Loads the PDF from cache, extracts text from the specified page using OCR (or native fallback), and returns it."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(CACHE_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found in cache. Please upload it first.")

    try:
        if use_ocr:
            try:
                import fitz
                from PIL import Image

                with fitz.open(file_path) as doc:
                    if page_index < 0 or page_index >= len(doc):
                        raise IndexError(f"Page index {page_index} out of range (0-{len(doc)-1})")
                    page = doc[page_index]
                    w = page.rect.width
                    h = page.rect.height
                    zoom = 2.0
                    mat = fitz.Matrix(zoom, zoom)

                    try:
                        import pytesseract
                        if w > h:
                            clip_left = fitz.Rect(0, 0, w / 2, h)
                            pix_left = page.get_pixmap(matrix=mat, clip=clip_left)
                            img_left = Image.open(io.BytesIO(pix_left.tobytes("png")))
                            text_left = pytesseract.image_to_string(img_left, lang='nep+jpn+eng')

                            clip_right = fitz.Rect(w / 2, 0, w, h)
                            pix_right = page.get_pixmap(matrix=mat, clip=clip_right)
                            img_right = Image.open(io.BytesIO(pix_right.tobytes("png")))
                            text_right = pytesseract.image_to_string(img_right, lang='nep+jpn+eng')

                            text = f"{text_left.strip()}\n\n{text_right.strip()}"
                        else:
                            pix = page.get_pixmap(matrix=mat)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            text = pytesseract.image_to_string(img, lang='nep+jpn+eng').strip()
                    except Exception:
                        # Fallback to winocr for local Windows debugging
                        import winocr
                        import concurrent.futures
                        
                        # Detect language from file_path
                        doc_path = file_path.lower() if 'file_path' in locals() else ""
                        lang = 'en'
                        import re
                        if any(k in doc_path for k in ['nihongo', 'jpn', 'japanese', 'n3', 'moji', 'goi']) or re.search(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf]', doc_path):
                            lang = 'ja'
                            
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            if w > h:
                                clip_left = fitz.Rect(0, 0, w / 2, h)
                                pix_left = page.get_pixmap(matrix=mat, clip=clip_left)
                                img_left = Image.open(io.BytesIO(pix_left.tobytes("png")))
                                res_left = executor.submit(winocr.recognize_pil_sync, img_left, lang).result()
                                text_left = res_left.get("text", "")

                                clip_right = fitz.Rect(w / 2, 0, w, h)
                                pix_right = page.get_pixmap(matrix=mat, clip=clip_right)
                                img_right = Image.open(io.BytesIO(pix_right.tobytes("png")))
                                res_right = executor.submit(winocr.recognize_pil_sync, img_right, lang).result()
                                text_right = res_right.get("text", "")

                                text = f"{text_left.strip()}\n\n{text_right.strip()}"
                            else:
                                pix = page.get_pixmap(matrix=mat)
                                img = Image.open(io.BytesIO(pix.tobytes("png")))
                                res = executor.submit(winocr.recognize_pil_sync, img, lang).result()
                                text = res.get("text", "").strip()

                    return {"text": text}
            except Exception as ocr_err:
                print(f"OCR failed: {ocr_err}. Falling back to native text extraction.")

        # Fallback / Native Text Extraction
        import fitz
        with fitz.open(file_path) as doc:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(f"Page index {page_index} out of range (0-{len(doc)-1})")
            page = doc[page_index]
            text = page.get_text("text")
            return {"text": text}

    except IndexError as idx_err:
        raise HTTPException(status_code=400, detail=str(idx_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while extracting page text: {str(e)}")


class TranslateRequest(BaseModel):
    text: str

@app.post("/form-filler/translate-nepali", tags=["Form Filler"])
def translate_nepali(req: TranslateRequest):
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='ne')
        translated = translator.translate(req.text)
        return {"translated": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/form-filler/upload", response_model=FormFillerUploadResponse, tags=["Form Filler"])
async def form_filler_upload(file: UploadFile = File(...)):
    """
    Uploads a PDF, runs field detection (digital, visual, character-grid),
    generates direct user-friendly questions, and caches the PDF file.
    """
    try:
        # Save file to cache (converting image to PDF if necessary)
        dest_path, filename = await _process_upload_to_pdf(file)

        # Run field detection in background thread
        detected_fields = await run_in_threadpool(detect_fields, dest_path)
        
        # Populate history
        total_pages = get_pdf_page_count(dest_path)
        file_size_bytes = os.path.getsize(dest_path)
        _add_to_history(filename, total_pages, file_size_bytes)

        # Generate questions
        fields_response = []
        for f in detected_fields:
            ctx_lower = str(f.get("context", "")).lower()
            name_lower = str(f.get("field_name", "")).lower()
            
            # Skip signature fields entirely. The user will sign them manually with a pen after printing.
            if "signature" in ctx_lower or "sign" in ctx_lower or "हस्ताक्षर" in ctx_lower or "signature" in name_lower or "sign" in name_lower:
                continue
                
            q = generate_question(f.get("field_name"), f.get("context"))
            fields_response.append(
                FormField(
                    id=f["id"],
                    type=f["type"],
                    page_index=f["page_index"],
                    rect=f["rect"],
                    cells=f.get("cells"),
                    context=f["context"],
                    question=q
                )
            )

        resp = FormFillerUploadResponse(filename=filename, fields=fields_response)
        
        # Save cache
        json_cache_path = os.path.join(CACHE_DIR, filename + ".fields.json")
        try:
            with open(json_cache_path, "w", encoding="utf-8") as f:
                f.write(resp.model_dump_json())
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return resp

    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing form filler upload: {str(e)}")


@app.get("/form-filler/fields", response_model=FormFillerUploadResponse, tags=["Form Filler"])
async def form_filler_get_fields(
    filename: str = Query(..., description="Name of the cached PDF file to detect fields from")
):
    """
    Detects fillable fields (AcroForm widgets, underlines, boxes, character grids)
    in an already-cached PDF and returns them with generated questions.

    Called by the frontend when opening a PDF from history (no re-upload needed).
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(CACHE_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found in cache. Please upload it first.")

    json_cache_path = os.path.join(CACHE_DIR, safe_filename + ".fields.json")
    if os.path.exists(json_cache_path):
        try:
            with open(json_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return FormFillerUploadResponse(**data)
        except Exception:
            pass

    try:
        detected_fields = await run_in_threadpool(detect_fields, file_path)

        fields_response = []
        for f in detected_fields:
            q = generate_question(f.get("field_name"), f.get("context"))
            fields_response.append(
                FormField(
                    id=f["id"],
                    type=f["type"],
                    page_index=f["page_index"],
                    rect=f["rect"],
                    cells=f.get("cells"),
                    context=f["context"],
                    question=q
                )
            )

        resp = FormFillerUploadResponse(filename=safe_filename, fields=fields_response)
        
        # Save cache
        try:
            with open(json_cache_path, "w", encoding="utf-8") as f:
                f.write(resp.model_dump_json())
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting fields: {str(e)}")





@app.post("/form-filler/fill", tags=["Form Filler"])
async def form_filler_fill(request: FormFillerFillRequest):
    """
    Fills visual blanks, character-grid cells, and digital AcroForm fields in the specified PDF.
    Applies custom pen color and returns the resulting PDF file.
    """
    safe_filename = os.path.basename(request.filename)
    input_path = os.path.join(CACHE_DIR, safe_filename)
    
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="PDF file not found in cache. Please upload it first.")

    # Create a temporary output file path
    output_filename = f"filled_{datetime.now().timestamp()}_{safe_filename}"
    output_path = os.path.join(CACHE_DIR, output_filename)

    try:
        # Convert Pydantic schemas to standard dictionaries for pdf_writer
        answers_dict = [{"field_id": a.field_id, "value": a.value} for a in request.answers]
        
        # Fill PDF
        await run_in_threadpool(
            fill_pdf,
            input_pdf_path=input_path,
            output_pdf_path=output_path,
            answers=answers_dict,
            pen_color=request.pen_color
        )

        # Open and read filled PDF content to stream back
        with open(output_path, "rb") as f:
            pdf_data = f.read()

        # Clean up temporary output file
        try:
            os.remove(output_path)
        except Exception:
            pass

        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=filled_{safe_filename}"}
        )

    except Exception as e:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error filling PDF: {str(e)}")


# ---------------------------------------------------------------------------
# User Profile endpoints
# ---------------------------------------------------------------------------

@app.get("/form-filler/profile", response_model=ProfileResponse, tags=["Form Filler"])
async def get_profile():
    """
    Returns the saved user profile key-value pairs.
    Used by the frontend to pre-populate the My Profile panel.
    """
    entries_raw = _profile_store.load()
    return ProfileResponse(
        entries=[ProfileEntry(key=e["key"], value=e["value"]) for e in entries_raw]
    )


@app.post("/form-filler/profile", response_model=ProfileResponse, tags=["Form Filler"])
async def save_profile(request: ProfileSaveRequest):
    """
    Saves (or replaces) the user profile with the provided key-value entries.
    The profile is persisted to disk as JSON inside the cache directory.
    """
    entries_dict = [{"key": e.key, "value": e.value} for e in request.entries]
    saved = _profile_store.save(entries_dict)
    return ProfileResponse(
        entries=[ProfileEntry(key=e["key"], value=e["value"]) for e in saved]
    )


@app.post("/form-filler/suggest", response_model=SuggestResponse, tags=["Form Filler"])
async def suggest_field_values(request: SuggestRequest):
    """
    Given a list of detected form field contexts (labels), semantically matches
    each one against the stored user profile and returns the best-matching value.

    Uses vector embeddings (Ollama nomic-embed-text) when available, or falls
    back to n-gram cosine similarity when Ollama is offline.
    """
    field_contexts = [
        {"field_id": f.field_id, "context": f.context}
        for f in request.fields
    ]
    raw_suggestions = _profile_store.suggest(field_contexts)
    return SuggestResponse(
        suggestions=[
            SuggestionResult(
                field_id=s["field_id"],
                suggested_value=s["suggested_value"],
                matched_key=s["matched_key"],
                score=s["score"],
            )
            for s in raw_suggestions
        ]
    )


# Mount Flutter Web App build folder to serve directly from the server
WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "static"))
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
