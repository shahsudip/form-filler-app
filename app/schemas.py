from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class UploadResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded and cached PDF file")
    total_pages: int = Field(..., description="Total number of pages in the PDF file")

class PdfHistoryItem(BaseModel):
    filename: str = Field(..., description="Name of the PDF file in cache")
    total_pages: int = Field(..., description="Total number of pages")
    uploaded_at: str = Field(..., description="ISO 8601 timestamp when first uploaded")
    file_size_bytes: int = Field(..., description="File size in bytes")

class FormField(BaseModel):
    id: str = Field(..., description="Unique ID for the field")
    type: str = Field(..., description="Type of the field: 'digital' or 'visual'")
    page_index: int = Field(..., description="0-based page index")
    rect: List[float] = Field(..., description="Bounding box [x0, y0, x1, y1]")
    cells: Optional[List[List[float]]] = Field(None, description="List of bounding boxes for each cell in a character grid")
    context: str = Field(..., description="Label or context nearby the field")
    question: str = Field(..., description="User-friendly generated question")

class FormFillerUploadResponse(BaseModel):
    filename: str = Field(..., description="Cached PDF filename")
    fields: List[FormField] = Field(..., description="Detected fields and questions")

class AnswerItem(BaseModel):
    field_id: str = Field(..., description="ID of the field being answered")
    value: str = Field(..., description="Text value to fill")

class FormFillerFillRequest(BaseModel):
    filename: str = Field(..., description="Cached PDF filename")
    pen_color: str = Field("#000000", description="Hex code for the custom pen color")
    answers: List[AnswerItem] = Field(..., description="List of answers")


# ---------------------------------------------------------------------------
# User Profile + Auto-Suggest schemas
# ---------------------------------------------------------------------------

class ProfileEntry(BaseModel):
    key: str = Field(..., description="Human-readable label for the profile field (e.g. 'Full Name')")
    value: str = Field(..., description="The stored value for this profile field")

class ProfileResponse(BaseModel):
    entries: List[ProfileEntry] = Field(..., description="All stored profile key-value pairs")

class ProfileSaveRequest(BaseModel):
    entries: List[ProfileEntry] = Field(..., description="Profile entries to save (replaces existing profile)")

class SuggestRequestItem(BaseModel):
    field_id: str = Field(..., description="Unique field ID from field detection")
    context: str = Field(..., description="Context label detected near the field in the PDF")

class SuggestRequest(BaseModel):
    fields: List[SuggestRequestItem] = Field(..., description="List of detected fields to get suggestions for")

class SuggestionResult(BaseModel):
    field_id: str = Field(..., description="The field ID this suggestion is for")
    suggested_value: str = Field("", description="Best-matching profile value; empty if no match found")
    matched_key: str = Field("", description="The profile key that produced this match")
    score: float = Field(0.0, description="Similarity score between 0.0 and 1.0")

class SuggestResponse(BaseModel):
    suggestions: List[SuggestionResult] = Field(..., description="One suggestion per requested field")

