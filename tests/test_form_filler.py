import os
import sys
import tempfile
from fastapi.testclient import TestClient

# Add parent directory to sys.path to resolve app imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.form_filler.field_detector import detect_fields
from app.form_filler.vector_store import SimpleVectorStore, fallback_cosine_similarity
from app.form_filler.question_generator import generate_question, generate_fallback_question
from app.form_filler.pdf_writer import fill_pdf

TARGET_PDF = r"D:\sudip_software\[weeblibrary.wordpress.com]_Nihongo_Power_Drill_N3_Moji_Goi\Mobile_Banking_Alert_eBanking_Application_Form.pdf"

def test_field_detector():
    """Verify that field detector parses character grids and contexts correctly."""
    print("Running field detector tests...")
    assert os.path.exists(TARGET_PDF), f"Target PDF not found: {TARGET_PDF}"
    
    fields = detect_fields(TARGET_PDF)
    assert len(fields) > 0, "No fields were detected in the form."
    
    # Check for presence of character-grid fields
    grid_fields = [f for f in fields if f["type"] == "character-grid"]
    assert len(grid_fields) >= 4, f"Expected at least 4 character grids, found {len(grid_fields)}"
    
    # Check that labels match expected contexts
    contexts = [f["context"].lower() for f in grid_fields]
    print(f"Detected contexts: {contexts}")
    
    assert any("date" in ctx for ctx in contexts), "Date field was not found in contexts."
    assert any("name" in ctx for ctx in contexts), "Account Name field was not found in contexts."
    assert any("account" in ctx for ctx in contexts), "Primary Account No field was not found in contexts."
    assert any("mobile" in ctx or "phone" in ctx for ctx in contexts), "Mobile Number field was not found in contexts."

def test_vector_store():
    """Verify SimpleVectorStore functionality, similarity search, and fallback."""
    print("Running vector store tests...")
    vstore = SimpleVectorStore()
    # Force fallback mode to test deterministic local n-gram math
    vstore.use_fallback = True
    
    vstore.add_item("id_1", "Date of Birth", {"field": "dob"})
    vstore.add_item("id_2", "Primary Account Number", {"field": "acc_no"})
    vstore.add_item("id_3", "Mobile Number / Phone Number", {"field": "mobile"})
    
    res = vstore.search("account", limit=2)
    assert len(res) > 0
    assert res[0]["id"] == "id_2"
    assert res[0]["score"] > 0.0
    
    res_mobile = vstore.search("phone number", limit=1)
    assert res_mobile[0]["id"] == "id_3"

def test_question_generator():
    """Verify question generation maps clean user-friendly questions."""
    print("Running question generator tests...")
    # Test common mappings
    q1 = generate_fallback_question("dob", "Date of Birth :")
    assert "date of birth" in q1.lower()
    
    q2 = generate_fallback_question("mobile_number", "Mobile Number")
    assert "phone" in q2.lower() or "mobile" in q2.lower()
    
    # Test clean generation fallback
    q3 = generate_fallback_question("some_custom_field", "Preferred Contact Method")
    assert "Preferred Contact Method" in q3

def test_pdf_writer_and_api():
    """Verify PDF writer fills character-grids and FastAPI integration endpoints."""
    print("Running PDF writer and API integration tests...")
    client = TestClient(app)
    
    # Test /form-filler/upload
    with open(TARGET_PDF, "rb") as f:
        response = client.post(
            "/form-filler/upload",
            files={"file": (os.path.basename(TARGET_PDF), f, "application/pdf")}
        )
    
    assert response.status_code == 200, f"Upload failed: {response.text}"
    data = response.json()
    assert "filename" in data
    assert "fields" in data
    
    fields = data["fields"]
    assert len(fields) > 0
    
    # Map answers to the detected character grids
    answers = []
    for f in fields:
        f_type = f["type"]
        f_ctx = f["context"].lower()
        if f_type == "character-grid":
            if "date" in f_ctx:
                answers.append({"field_id": f["id"], "value": "16062026"})
            elif "name" in f_ctx:
                answers.append({"field_id": f["id"], "value": "JOHN DOE"})
            elif "account" in f_ctx:
                answers.append({"field_id": f["id"], "value": "12345678901234"})
            elif "mobile" in f_ctx:
                answers.append({"field_id": f["id"], "value": "98765432"})
    
    # Test /form-filler/fill
    fill_payload = {
        "filename": data["filename"],
        "pen_color": "#FF0000",
        "answers": answers
    }
    
    fill_response = client.post("/form-filler/fill", json=fill_payload)
    assert fill_response.status_code == 200, f"Fill failed: {fill_response.text}"
    assert fill_response.headers["content-type"] == "application/pdf"
    assert len(fill_response.content) > 0

if __name__ == "__main__":
    test_field_detector()
    test_vector_store()
    test_question_generator()
    test_pdf_writer_and_api()
    print("\nAll tests completed successfully!")
