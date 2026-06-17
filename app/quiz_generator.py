import re
import random
from typing import List
from app.schemas import QuizQuestion

STOPWORDS = {
    "the", "and", "a", "of", "to", "in", "is", "that", "it", "for", "on", "with", 
    "as", "this", "was", "at", "by", "an", "be", "are", "from", "their", "or", 
    "which", "but", "has", "have", "they", "not", "with", "about", "more", "also", 
    "its", "into", "than", "other", "some", "only", "them", "these", "then", 
    "would", "could", "should", "will", "can", "there", "their", "what", "which", 
    "who", "how", "where", "when", "why", "been", "were", "had", "has", "have"
}

def is_japanese(text: str) -> bool:
    """Detects if text contains Japanese Hiragana, Katakana, or Kanji characters."""
    for c in text:
        code = ord(c)
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            return True
    return False

def split_into_sentences_en(text: str) -> List[str]:
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    raw_sentences = sentence_end.split(text)
    sentences = []
    for s in raw_sentences:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if 40 <= len(s_clean) <= 250:
            sentences.append(s_clean)
    return sentences

def split_into_sentences_ja(text: str) -> List[str]:
    raw_sentences = re.split(r'[。？?\n\r]+', text)
    sentences = []
    for s in raw_sentences:
        s_clean = re.sub(r'\s+', '', s).strip()
        if 15 <= len(s_clean) <= 150:
            sentences.append(s_clean)
    return sentences

def extract_key_terms_en(text: str, num_terms: int = 20) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]{5,15}\b', text.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    sorted_terms = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    return sorted_terms[:num_terms]

def extract_key_terms_ja(text: str, num_terms: int = 30) -> List[str]:
    quoted_terms = re.findall(r'[「『]([^「」『』]{2,10})[」』]', text)
    katakana_terms = re.findall(r'[\u30a0-\u30ff]{2,8}', text)
    combined = list(set(quoted_terms + katakana_terms))
    if len(combined) < num_terms:
        kanji_terms = re.findall(r'[\u4e00-\u9fff]{2,5}', text)
        combined = list(set(combined + kanji_terms))
    return combined[:num_terms]

def generate_quiz(text: str, max_questions: int = 150) -> List[QuizQuestion]:
    # 1. Try to parse direct JLPT questions and options from the PDF text first
    direct_questions = parse_direct_jlpt_questions(text, max_questions)
    if len(direct_questions) >= 3:
        return direct_questions
        
    # 2. Fall back to automatic generation if direct questions are not found
    if is_japanese(text):
        return generate_quiz_ja(text, max_questions)
    return generate_quiz_en(text, max_questions)

def parse_direct_jlpt_questions(text: str, max_questions: int) -> List[QuizQuestion]:
    lines = text.split("\n")
    questions = []
    
    current_part = ""
    current_mondai = ""
    
    state = "LOOKING"
    q_text_buffer = []
    opt_text_buffer = []
    q_num = ""
    
    # Pre-clean lines to skip floating furigana lines
    cleaned_lines = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if line is floating furigana line
        # If it contains only hiragana/katakana/spaces/punctuation, and the next line starts with a question bracket
        is_furigana_only = all(ord(c) in [32, 12288] or (0x3040 <= ord(c) <= 0x309F) or (0x30A0 <= ord(c) <= 0x30FF) for c in line)
        if is_furigana_only and idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if re.match(r'^[［\[\(「]\d+[］\]\)」]', next_line):
                continue
                
        cleaned_lines.append(line)
        
    def clean_ocr_noise(s: str) -> str:
        # Remove known garbage patterns like "ittiv", combining accents, greek characters, and rare box font errors
        s = re.sub(r'\bittiv\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'[α-ω\u03b0-\u03ff]', '', s) # Greek letters
        s = re.sub(r'[\u0300-\u036f]', '', s) # combining accents
        s = re.sub(r'[ζλωτθπρσϕψ]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def save_current_question():
        nonlocal q_text_buffer, opt_text_buffer, q_num, current_part, current_mondai
        if not q_text_buffer or not opt_text_buffer:
            return
            
        q_text = " ".join(q_text_buffer).strip()
        opt_text = " ".join(opt_text_buffer).strip()
        
        opt_match = re.search(r'①\s*(.*?)\s*②\s*(.*?)\s*③\s*(.*?)\s*④\s*(.*)', opt_text)
        if opt_match:
            options = [
                opt_match.group(1).strip(),
                opt_match.group(2).strip(),
                opt_match.group(3).strip(),
                opt_match.group(4).strip()
            ]
            
            clean_q_text = re.sub(r'^[［\[\(「]\d+[］\]\)」]\s*', '', q_text)
            
            # Clean OCR/font-mapping noise from question and options
            clean_q_text = clean_ocr_noise(clean_q_text)
            options = [clean_ocr_noise(opt) for opt in options]
            
            # Basic validation: make sure options are not empty
            if all(len(opt) > 0 for opt in options):
                questions.append(QuizQuestion(
                    question_text=clean_q_text,
                    options=options,
                    correct_option_index=0, # default index in practice mode
                    explanation=f"【PDF直出し問題】この問題はPDFの「{current_part or '問題'} {current_mondai or ''}」からそのまま抽出されました。正しい解答は、お手元のテキストの解答用紙（解答欄）を確認してください。",
                    part_name=clean_ocr_noise(current_part),
                    mondai_number=clean_ocr_noise(current_mondai),
                    original_question_no=q_num
                ))
            
        q_text_buffer = []
        opt_text_buffer = []
        q_num = ""

    for line in cleaned_lines:
        if len(questions) >= max_questions:
            break
            
        # Detect part headers
        part_match = re.search(r'(言語知識|文字・語彙|文法|読解|聴解|語彙知識)', line)
        if part_match and "問題" not in line and "［" not in line and "[" not in line and "①" not in line:
            clean_part = line.replace("---", "").strip()
            clean_part = re.sub(r'PAGE \d+ (LEFT|RIGHT)', '', clean_part)
            clean_part = re.sub(r'\(Two-Col:.*?\)', '', clean_part).strip()
            if clean_part:
                current_part = clean_part
            continue
            
        # Detect mondai headers
        mondai_match = re.match(r'^(問題[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ\d]+|問\d+)', line)
        if mondai_match:
            current_mondai = line.split("：")[0].split(":")[0].strip()
            continue
            
        # Detect question start
        q_start_match = re.match(r'^([［\[\(「](?P<num>\d+)[］\]\)」])', line)
        if q_start_match:
            save_current_question()
            q_num = q_start_match.group("num")
            q_text_buffer = [line]
            state = "QUESTION"
            continue
            
        if state == "QUESTION":
            if "①" in line:
                opt_text_buffer = [line]
                state = "OPTIONS"
            else:
                q_text_buffer.append(line)
        elif state == "OPTIONS":
            if "①" in line or "②" in line or "③" in line or "④" in line:
                opt_text_buffer.append(line)
            elif len(opt_text_buffer) < 4:
                opt_text_buffer.append(line)
                
    save_current_question()
    return questions

def generate_quiz_ja(text: str, max_questions: int) -> List[QuizQuestion]:
    sentences = split_into_sentences_ja(text)
    key_terms = extract_key_terms_ja(text, num_terms=40)
    
    fallback_terms = ["日本語", "語彙", "文法", "練習", "試験", "理解", "言葉", "会話", "単語", "意味"]
    for ft in fallback_terms:
        if len(key_terms) >= 30:
            break
        if ft not in key_terms:
            key_terms.append(ft)
            
    questions = []
    used_sentences = set()
    
    def_pattern = re.compile(r'([^\s「」]{2,10})とは、?([^\s。]+)(のこと|をいう|である|です)', re.IGNORECASE)
    
    for sentence in sentences:
        if len(questions) >= max_questions:
            break
        match = def_pattern.search(sentence)
        if match:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            if len(definition) > 6:
                distractors = [t for t in key_terms if t != term][:15]
                if len(distractors) < 3:
                    continue
                selected_distractors = random.sample(distractors, 3)
                options = [term] + selected_distractors
                random.shuffle(options)
                correct_idx = options.index(term)
                
                q = QuizQuestion(
                    question_text=f"次の説明にあてはまる正しい言葉を選んでください：\n「{definition}」",
                    options=options,
                    correct_option_index=correct_idx,
                    explanation=f"本文の記述：「{sentence}」"
                )
                questions.append(q)
                used_sentences.add(sentence)
                
    for sentence in sentences:
        if len(questions) >= max_questions:
            break
        if sentence in used_sentences:
            continue
            
        matching_terms = []
        for term in key_terms:
            if term in sentence:
                matching_terms.append(term)
                
        if not matching_terms:
            continue
            
        target_term = max(matching_terms, key=len)
        sentence_blanked = sentence.replace(target_term, "________")
        
        distractors = [t for t in key_terms if t != target_term][:15]
        if len(distractors) < 3:
            continue
        selected_distractors = random.sample(distractors, 3)
        options = [target_term] + selected_distractors
        random.shuffle(options)
        correct_idx = options.index(target_term)
        
        q = QuizQuestion(
            question_text=f"空欄に入る最も適切な言葉を選んでください：\n「{sentence_blanked}」",
            options=options,
            correct_option_index=correct_idx,
            explanation=f"本文의記述：「{sentence}」"
        )
        questions.append(q)
        used_sentences.add(sentence)
        
    return questions

def generate_quiz_en(text: str, max_questions: int) -> List[QuizQuestion]:
    sentences = split_into_sentences_en(text)
    key_terms = extract_key_terms_en(text, num_terms=30)
    
    fallback_terms = ["system", "process", "concept", "structure", "theory", "method", "analysis", "function", "development", "element"]
    for ft in fallback_terms:
        if len(key_terms) >= 30:
            break
        if ft not in key_terms:
            key_terms.append(ft)
            
    questions = []
    used_sentences = set()
    
    def_pattern = re.compile(
        r'\b([A-Z][a-zA-Z0-9\s-]{2,20})\s+(is|are|refers\s+to|means|is\s+defined\s+as)\s+(.+)', 
        re.IGNORECASE
    )
    
    for sentence in sentences:
        if len(questions) >= max_questions:
            break
        match = def_pattern.search(sentence)
        if match:
            term = match.group(1).strip()
            definition = match.group(3).strip()
            if definition.endswith('.'):
                definition = definition[:-1]
                
            if len(term.split()) <= 3 and len(definition) > 15:
                distractors = [t for t in key_terms if t.lower() != term.lower()][:10]
                if len(distractors) < 3:
                    continue
                selected_distractors = random.sample(distractors, 3)
                term_cased = term.title() if term[0].isupper() else term.lower()
                options = [term_cased] + [d.title() if term[0].isupper() else d.lower() for d in selected_distractors]
                random.shuffle(options)
                correct_idx = options.index(term_cased)
                
                q = QuizQuestion(
                    question_text=f"Which term is defined as: '{definition}'?",
                    options=options,
                    correct_option_index=correct_idx,
                    explanation=f"Based on the text: '{sentence}'"
                )
                questions.append(q)
                used_sentences.add(sentence)
                
    for sentence in sentences:
        if len(questions) >= max_questions:
            break
        if sentence in used_sentences:
            continue
            
        matching_terms = []
        for term in key_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', sentence, re.IGNORECASE):
                matching_terms.append(term)
                
        if not matching_terms:
            continue
            
        target_term = max(matching_terms, key=len)
        pattern = re.compile(r'\b' + re.escape(target_term) + r'\b', re.IGNORECASE)
        match_in_text = pattern.search(sentence)
        if not match_in_text:
            continue
        term_actual_case = match_in_text.group(0)
        sentence_blanked = pattern.sub("________", sentence)
        
        distractors = [t for t in key_terms if t.lower() != target_term.lower()][:10]
        if len(distractors) < 3:
            continue
        selected_distractors = random.sample(distractors, 3)
        options = [term_actual_case]
        for d in selected_distractors:
            if term_actual_case.istitle():
                options.append(d.title())
            elif term_actual_case.isupper():
                options.append(d.upper())
            else:
                options.append(d.lower())
                
        random.shuffle(options)
        correct_idx = options.index(term_actual_case)
        
        q = QuizQuestion(
            question_text=f"Fill in the blank: '{sentence_blanked}'",
            options=options,
            correct_option_index=correct_idx,
            explanation=f"The text states: '{sentence}'"
        )
        questions.append(q)
        used_sentences.add(sentence)
        
    return questions
