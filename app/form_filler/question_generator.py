import re
import requests

# Dictionary of standard fallbacks for common fields
COMMON_FIELD_MAPPINGS = {
    "name": "What is your full name?",
    "first_name": "What is your first name?",
    "last_name": "What is your last name?",
    "given_name": "What is your first name?",
    "family_name": "What is your last name?",
    "dob": "What is your date of birth?",
    "date_of_birth": "What is your date of birth?",
    "birth_date": "What is your date of birth?",
    "address": "What is your address?",
    "street": "What is your street address?",
    "city": "What is your city?",
    "state": "What is your state?",
    "zip": "What is your ZIP code?",
    "zip_code": "What is your ZIP code?",
    "postal_code": "What is your postal code?",
    "phone": "What is your phone number?",
    "phone_number": "What is your phone number?",
    "telephone": "What is your telephone number?",
    "mobile": "What is your mobile number?",
    "email": "What is your email address?",
    "email_address": "What is your email address?",
    "date": "What is the date?",
    "signature": "Please sign here.",
    "sign": "Please sign here.",
    "gender": "What is your gender?",
    "sex": "What is your gender?",
    "title": "What is your title?",
    "company": "What is your company name?",
    "occupation": "What is your occupation?",
    "job_title": "What is your job title?"
}

# Add more specific phrases to COMMON_FIELD_MAPPINGS
COMMON_FIELD_MAPPINGS["date of birth"] = "What is your date of birth?"
COMMON_FIELD_MAPPINGS["birth"] = "What is your date of birth?"

def generate_fallback_question(field_name: str, context: str) -> str:
    """Generates a friendly question using rule-based parsing as a fallback."""
    # Prioritize context label if it contains useful words
    clean_ctx = context.strip() if context else ""
    # Strip underscores, colons, spaces from context
    clean_ctx = re.sub(r"__+", "", clean_ctx)
    clean_ctx = clean_ctx.strip().rstrip(":").rstrip(" ").rstrip("-")
    
    # Check if context matches common patterns (longest keys first)
    ctx_lower = clean_ctx.lower()
    sorted_keys = sorted(COMMON_FIELD_MAPPINGS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        val = COMMON_FIELD_MAPPINGS[key]
        if key in ctx_lower or re.search(r"\b" + re.escape(key) + r"\b", ctx_lower):
            return val

    # If no context, fall back to field_name
    name_lower = field_name.lower() if field_name else ""
    name_lower = re.sub(r"_\d+$", "", name_lower)
    name_lower = re.sub(r"\d+$", "", name_lower)
    for key in sorted_keys:
        val = COMMON_FIELD_MAPPINGS[key]
        key_parts = key.replace("_", " ").replace("-", " ").split()
        if key == name_lower or any(part in name_lower.split("_") or part in name_lower.split(" ") for part in key_parts):
            return val

    # General conversion:
    # If we have a good context, build "What is your [context]?"
    if clean_ctx and len(clean_ctx) > 1:
        # Avoid double 'what' or 'please'
        if clean_ctx.lower().startswith("what") or clean_ctx.lower().startswith("please") or clean_ctx.lower().startswith("enter"):
            return clean_ctx if clean_ctx.endswith("?") or clean_ctx.endswith(".") else clean_ctx + "?"
        return f"Please fill in: {clean_ctx}"
        
    # If no context but we have a name
    if field_name:
        # Convert snake_case or camelCase to spaces
        spaced_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", field_name)
        spaced_name = spaced_name.replace("_", " ").replace("-", " ")
        spaced_name = spaced_name.strip().capitalize()
        return f"Please enter: {spaced_name}"
        
    return "Please fill in this field."

def generate_question(field_name: str, context: str, model_name: str = "llama3", ollama_url: str = "http://localhost:11434") -> str:
    """
    Generates a natural user-friendly question from field name and context.
    Uses rule-based generation to minimize AI usage and ensure instantaneous response.
    """
    return generate_fallback_question(field_name, context)
