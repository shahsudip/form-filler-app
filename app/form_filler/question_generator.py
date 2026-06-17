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
    "job_title": "What is your job title?",
    "account holder's nepali name": "What is the account holder's Nepali name?",
    "account holder nepali name": "What is the account holder's Nepali name?",
    "name in nepali": "What is the account holder's Nepali name?",
    "nepali name": "What is the account holder's Nepali name?",
    "account holder's name in nepali": "What is the account holder's Nepali name?",
    "account holder name in nepali": "What is the account holder's Nepali name?",
    "name (in nepali)": "What is the account holder's Nepali name?",
    "name in devanagari": "What is the account holder's Nepali name?",
    "name (in devanagari)": "What is the account holder's Nepali name?"
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
    
    # Translate context to English if it contains non-English characters
    try:
        from deep_translator import GoogleTranslator
        # Detect any non-ASCII characters (like Japanese or Nepali)
        if clean_ctx and re.search(r'[^\x00-\x7F]', clean_ctx):
            clean_ctx = GoogleTranslator(source='auto', target='en').translate(clean_ctx)
    except Exception as e:
        print(f"Translation failed: {e}")
    
    # Check for direct Japanese matches (using unicode escape sequences to prevent source file corruption)
    ctx_lower = clean_ctx.lower()
    if "\u6c0f\u540d" in ctx_lower or "\u304a\u540d\u524d" in ctx_lower:
        return "What is your full name?"
    if "\u751f\u5e74\u6708\u65e5" in ctx_lower:
        return "What is your date of birth?"
    if "\u4f4f\u6240" in ctx_lower:
        return "What is your address?"
    if "\u96fb\u8a71\u756a\u53f7" in ctx_lower or "\u9023\u7d61\u5148" in ctx_lower:
        return "What is your phone number?"
    if "\u53e3\u5ea7\u756a\u53f7" in ctx_lower:
        return "What is the account number?"
    if "\u30d5\u30ea\u30ac\u30ca" in ctx_lower or "\u3075\u308a\u304c\u306a" in ctx_lower:
        return "What is the pronunciation (Furigana)?"

    # Check if context matches common patterns (longest keys first)
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
        # If it still contains non-ASCII characters (failed translation), do not output Japanese to the chat!
        if re.search(r'[^\x00-\x7F]', clean_ctx):
            if field_name and not re.search(r'[^\x00-\x7F]', field_name):
                spaced_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", field_name)
                spaced_name = spaced_name.replace("_", " ").replace("-", " ")
                spaced_name = spaced_name.strip().capitalize()
                return f"Please enter: {spaced_name}"
            return "Please fill in this field."
            
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
