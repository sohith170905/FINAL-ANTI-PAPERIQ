import heapq
import os
import re
import time
import json
import hashlib
import uuid
import io
import html
import math
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
import docx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
import pandas as pd

# Import authentication and persistent database management
import auth
import summarizer

# ----------------------------------------------------
# NLTK Resource Initialization (Streamlit Cloud Safe)
# ----------------------------------------------------
for _resource in ("punkt_tab", "punkt", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="PaperIQ – AI Research Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
# Centralized Score Calculation & Verification
# ----------------------------------------------------
def calculate_overall_score(language, coherence, argumentation, academic_style, readability):
    """
    Authoritative single calculation for PaperIQ overall composite score.
    Weights:
      Language Quality: 25%
      Structural Coherence: 25%
      Argumentation: 20%
      Academic Style: 15%
      Readability: 15%
    """
    def _validate_score(val):
        if val is None:
            return 0.0
        try:
            return max(0.0, min(100.0, float(val)))
        except (TypeError, ValueError):
            return 0.0

    lang_val = _validate_score(language)
    coh_val = _validate_score(coherence)
    arg_val = _validate_score(argumentation)
    sty_val = _validate_score(academic_style)
    rd_val = _validate_score(readability)

    comp = (
        lang_val * 0.25 +
        coh_val * 0.25 +
        arg_val * 0.20 +
        sty_val * 0.15 +
        rd_val * 0.15
    )
    return round(max(0.0, min(100.0, comp)), 1)

def get_overall_score(res_or_scores):
    """
    Authoritative single getter for overall score across all pages, reports, and DB records.
    Guarantees no conflicting score values anywhere in the application.
    """
    if res_or_scores is None:
        return 0.0
    if isinstance(res_or_scores, (int, float)):
        try:
            return round(max(0.0, min(100.0, float(res_or_scores))), 1)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(res_or_scores, dict):
        if "overall_score" in res_or_scores and res_or_scores["overall_score"] is not None:
            try:
                return round(float(res_or_scores["overall_score"]), 1)
            except (TypeError, ValueError):
                pass
        if "score" in res_or_scores and res_or_scores["score"] is not None:
            try:
                return round(float(res_or_scores["score"]), 1)
            except (TypeError, ValueError):
                pass
        scores = res_or_scores.get("scores", res_or_scores)
        if isinstance(scores, dict):
            if "overall_score" in scores and scores["overall_score"] is not None:
                try:
                    return round(float(scores["overall_score"]), 1)
                except (TypeError, ValueError):
                    pass
            if "Composite" in scores and scores["Composite"] is not None:
                try:
                    return round(float(scores["Composite"]), 1)
                except (TypeError, ValueError):
                    pass
            # If component scores are present, calculate authoritative weighted score
            if any(k in scores for k in ("Language Quality", "Language", "Structural Coherence", "Coherence", "language_quality", "structural_coherence")):
                lang = scores.get("Language Quality", scores.get("Language", scores.get("language_quality", 0.0)))
                coh = scores.get("Structural Coherence", scores.get("Coherence", scores.get("structural_coherence", 0.0)))
                arg = scores.get("Argumentation", scores.get("Reasoning", scores.get("argumentation", 0.0)))
                sty = scores.get("Academic Style", scores.get("Sophistication", scores.get("academic_style", 0.0)))
                rd = scores.get("Readability", scores.get("readability", 0.0))
                return calculate_overall_score(lang, coh, arg, sty, rd)
    return 0.0

# ----------------------------------------------------
# Centralized Filename Normalizer & Title Resolver
# ----------------------------------------------------
def get_clean_base_filename(filename):
    """
    Sanitizes raw filenames, strips path traversal, duplicate extensions (.pdf_report.pdf),
    embedded report headers ('PaperIQ Analysis Report: ...'), score/grade fragments, and copy numbers.
    Returns the core manuscript identifier.
    """
    if not filename:
        return "Research_Paper"
    raw = str(filename).strip()
    
    # If the string contains an embedded file name like Gangadharan...pdf, extract it first!
    m_file = re.search(r"([A-Za-z0-9_\-\.]{4,}\.(?:pdf|docx|txt))", raw, re.I)
    if m_file:
        raw = m_file.group(1)
        
    # Strip any report/score clutter
    raw = re.sub(r"^(paperiq\s+analysis\s+report\s*[:\-–—]?\s*)+", "", raw, flags=re.I)
    raw = re.sub(r"^(paperiq[_\-\s]+report[_\-\s]*)+", "", raw, flags=re.I)
    raw = re.sub(r"^(report[_\-\s]*)+", "", raw, flags=re.I)
    raw = re.sub(r"(final\s+)?composite\s+score\s*[:\-–—]?\s*[\d.]+(\s*[\/\\]\s*\d+)?", "", raw, flags=re.I)
    raw = re.sub(r"score\s*[:\-–—]?\s*[\d.]+(\s*[\/\\]\s*\d+)?", "", raw, flags=re.I)
    raw = re.sub(r"grade\s*[:\-–—]?\s*[A-F][+\-]?", "", raw, flags=re.I)
    
    # Iteratively strip duplicate extensions and report suffixes
    base = raw
    changed = True
    while changed:
        changed = False
        lower = base.lower().strip()
        for ext in (".pdf", ".docx", ".txt", ".doc"):
            if lower.endswith(ext):
                base = base.strip()[:-len(ext)]
                lower = base.lower().strip()
                changed = True
        m = re.search(r"([_\-.]?(paperiq[_\-]?report|report))([_\-]?\d+)?$", base, re.I)
        if m:
            base = base[:m.start()]
            changed = True
            
    clean = re.sub(r"[^\w\-.]", "_", base).strip(" ._-")
    clean = re.sub(r"_+", "_", clean)
    if not clean or clean.lower() in ("report", "paperiq", "analysis", "paperiq_report", "research_paper"):
        clean = "Research_Paper"
    return clean

def format_clean_display_title(base):
    """Formats raw filename base into a professional human-readable title without generated suffixes."""
    if not base or base == "Research_Paper":
        return "Research Paper"
    # Match author-year code patterns like Gangadharan45122023JEAI111549 -> Gangadharan et al. – Research Paper
    m = re.match(r"^([A-Z][a-z]{3,})\d+[A-Za-z0-9]*$", base)
    if m:
        return f"{m.group(1)} et al. – Research Paper"
    # Replace underscores and hyphens with spaces
    spaced = base.replace("_", " ").replace("-", " ")
    # Clean up double spaces
    spaced = re.sub(r"\s+", " ", spaced).strip()
    if spaced.isupper() or spaced.islower():
        spaced = spaced.title()
    return spaced

def get_clean_report_filename(filename):
    """
    Guarantees strictly formatted: PaperIQ_Report_<clean_slug>.pdf with only ONE prefix/suffix.
    Never produces repeated suffixes (e.g. .pdf_report.pdf_Report.pdf).
    """
    clean_base = get_clean_base_filename(filename)
    m = re.match(r"^([A-Z][a-z]{3,})\d+[A-Za-z0-9]*$", clean_base)
    if m:
        slug = f"{m.group(1)}_et_al"
    else:
        slug = re.sub(r"[^\w\-]", "_", clean_base).strip("_")
        slug = re.sub(r"_+", "_", slug)
    if not slug or slug.lower() in ("report", "paperiq", "analysis", "paperiq_report", "research_paper"):
        slug = "Research_Paper"
    return f"PaperIQ_Report_{slug}.pdf"

# ----------------------------------------------------
# Multi-Format Text Extraction (PDF • DOCX • TXT)
# ----------------------------------------------------
def clean_extracted_text(text):
    """
    Cleans raw extracted text: removes header/footer noise, page numbers,
    reconnects hyphenated line-breaks, normalizes whitespace, and preserves
    citations, equations, and references.
    """
    if not text:
        return ""
        
    # Reconnect hyphenated words split across line breaks (e.g., "trans-\nformer" -> "transformer")
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)
    
    # Normalize non-breaking and irregular spaces
    text = re.sub(r"[\u00a0\u200b\u2009\u202f]", " ", text)
    
    # Repair missing spaces after punctuation abutting words (e.g., "model.We" -> "model. We")
    text = re.sub(r"([a-zA-Z]{2,})\.([A-Z][a-z])", r"\1. \2", text)
    text = re.sub(r"([a-zA-Z]{2,}),([a-zA-Z])", r"\1, \2", text)
    text = re.sub(r"([a-zA-Z]{2,}):([A-Z])", r"\1: \2", text)
    text = re.sub(r"([a-zA-Z]{2,});([A-Za-z])", r"\1; \2", text)
    text = re.sub(r"([a-zA-Z])(\d+[\d\.]*%)", r"\1 \2", text)
    
    lines = text.splitlines()
    cleaned_lines = []
    
    noise_patterns = [
        re.compile(r"^\d+\s*$", re.IGNORECASE),
        re.compile(r"^page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^volume\s*\d+.*$", re.IGNORECASE),
        re.compile(r"^authorized\s+licensed\s+use.*$", re.IGNORECASE),
        re.compile(r"^ieee\s+transactions.*$", re.IGNORECASE),
        re.compile(r"^\d{4,}\s+volume\s*\d+.*$", re.IGNORECASE),
        re.compile(r"^acm\s+transactions.*$", re.IGNORECASE),
        re.compile(r"^arxiv:\d+\.\d+.*$", re.IGNORECASE),
        re.compile(r"^doi:\s*10\.\d+.*$", re.IGNORECASE),
        re.compile(r"^issn:\s*\d+.*$", re.IGNORECASE),
        re.compile(r"^©\s*\d{4}.*$", re.IGNORECASE),
        re.compile(r"^all\s+rights\s+reserved.*$", re.IGNORECASE)
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        # Filter obvious single-line header/footer noise
        if len(stripped) < 90 and any(p.match(stripped) for p in noise_patterns):
            continue
        # Strip control / null characters while preserving normal unicode
        stripped = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", stripped)
        cleaned_lines.append(stripped)
    
    assembled = "\n".join(cleaned_lines)
    assembled = re.sub(r"\n{3,}", "\n\n", assembled)
    return assembled.strip()

def extract_text_from_file(file):
    """
    Robust multi-format extraction for PDF, DOCX, and TXT documents.
    Detects scanned image-only PDFs, password encryption, and file limits.
    """
    filename = getattr(file, "name", "document.pdf").lower()
    text = ""
    is_scanned = False
    
    # Check file size limit (30 MB)
    if hasattr(file, "size") and file.size > 30 * 1024 * 1024:
        return "__ERROR_FILE_TOO_LARGE__"
        
    try:
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
                
        if filename.endswith(".pdf"):
            try:
                with pdfplumber.open(file) as pdf:
                    total_images = 0
                    for page in pdf.pages:
                        extracted = page.extract_text(x_tolerance=2, y_tolerance=3, layout=False)
                        if not extracted:
                            extracted = page.extract_text(layout=True)
                        if extracted:
                            text += extracted + "\n\n"
                        if getattr(page, "images", None):
                            total_images += len(page.images)
                    if not text.strip() and total_images > 0:
                        is_scanned = True
            except Exception as pdf_err:
                err_str = str(pdf_err).lower()
                if any(w in err_str for w in ["password", "encrypted", "decrypt"]):
                    return "__ERROR_PASSWORD_PROTECTED__"
                return "__ERROR_CORRUPTED_PDF__"
                
            if is_scanned and not text.strip():
                return "__ERROR_SCANNED_PDF__"
                
        elif filename.endswith(".docx"):
            try:
                doc = docx.Document(file)
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                # Also extract text from tables
                table_text = []
                for table in doc.tables:
                    for row in table.rows:
                        row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_vals:
                            table_text.append(" | ".join(row_vals))
                combined_docx = paragraphs + table_text
                text = "\n\n".join(combined_docx)
            except Exception:
                return "__ERROR_CORRUPTED_DOCX__"
                
        elif filename.endswith(".txt"):
            content = file.getvalue() if hasattr(file, "getvalue") else (file.read() if hasattr(file, "read") else file)
            if isinstance(content, bytes):
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        text = content.decode(enc)
                        break
                    except Exception:
                        pass
                if not text:
                    text = content.decode("utf-8", errors="ignore")
            else:
                text = str(content)
                
    except Exception:
        return ""
    finally:
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
                
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return "__ERROR_EMPTY_DOCUMENT__"
    return cleaned

def extract_text_from_pdf(file):
    """Backwards compatibility alias for PDF files."""
    return extract_text_from_file(file)

def clean_text(text):
    if not text:
        return ""
    return clean_extracted_text(text)

# ----------------------------------------------------
# Section Detection & Heading Cleaning Pipeline
# ----------------------------------------------------
def is_report_noise_line(line):
    if not line:
        return True
    s = str(line).strip().lower()
    report_markers = [
        "paperiq", "analysis report", "composite score", "writing & structure score",
        "document metadata", "document name:", "evaluation scores", "grade:",
        "academic writing tone", "reading difficulty", "potential research areas",
        "corpus & lexical", "responsible ai", "analytical scoring framework",
        "final composite score", "score:"
    ]
    return any(marker in s for marker in report_markers)

def is_journal_or_metadata_noise(line):
    s = line.strip().lower()
    if len(s) < 2:
        return True
    if re.match(r"^[\d\s.,\-]+$", s):
        return True
    noise_tokens = [
        "volume", "issue", "issn", "isbn", "doi:", "http:", "https:",
        "all rights reserved", "authorized licensed", "ieee", "acm", "springer",
        "elsevier", "conference on", "proceedings of", "transactions on",
        "copyright", "©", "published by", "pp.", "pages "
    ]
    return any(token in s for token in noise_tokens)

def extract_paper_title_from_text(text):
    """
    Extracts the actual scholarly paper title from the document text.
    Inspects initial lines before Abstract, skipping journal metadata, author blocks,
    and generated PaperIQ report headers.
    """
    if not text:
        return None
        
    # Check if the text is from an exported PaperIQ report
    # If so, extract the underlying document name rather than the report header.
    sample = text[:1800]
    m_docname = re.search(r"Document Name:\s*([^\n\r]+)", sample, re.I)
    if m_docname:
        clean_base = get_clean_base_filename(m_docname.group(1))
        return format_clean_display_title(clean_base)
        
    lines = [l.strip() for l in text.split("\n")[:30] if l.strip()]
    candidate_lines = []
    for line in lines:
        l_lower = line.lower()
        if any(l_lower.startswith(prefix) for prefix in [
            "abstract", "keywords", "key words", "1. introduction", "1 introduction", "i. introduction", "contents"
        ]):
            break
        if is_journal_or_metadata_noise(line) or is_report_noise_line(line):
            continue
        if any(x in l_lower for x in ["@", "http", "doi:", "issn", "isbn", "volume", "issue", "pages", "copyright", "all rights reserved"]):
            continue
        if any(aff in l_lower for aff in ["department of", "university", "institute of", "college", "faculty of", "school of", "center for", "laboratory"]):
            continue
        words = line.split()
        if len(words) < 3 or len(line) < 12:
            continue
        if 12 <= len(line) <= 160 and len(words) <= 22:
            candidate_lines.append(line)
            if len(candidate_lines) >= 2:
                break
    if candidate_lines:
        full_title = " ".join(candidate_lines).strip()
        full_title = re.sub(r"\s+", " ", full_title)
        if 15 <= len(full_title) <= 180 and len(full_title.split()) >= 3:
            if not is_report_noise_line(full_title):
                return full_title
    return None

def get_clean_display_title(filename, text=None, stored_title=None):
    """
    Central display title resolver following priority:
    1. Pre-stored or extracted actual paper title from manuscript (if free from report/score noise).
    2. Intelligent author/clean title from filename.
    Never exposes generated report suffixes, scores, or report clutter.
    """
    if stored_title:
        st_clean = str(stored_title).strip()
        if len(st_clean) > 5 and not is_report_noise_line(st_clean) and not re.search(r"score\s*:\s*[\d.]+", st_clean, re.I):
            return st_clean
    if text:
        extracted = extract_paper_title_from_text(text)
        if extracted and not is_report_noise_line(extracted):
            return extracted
    base = get_clean_base_filename(filename)
    return format_clean_display_title(base)

def load_analysis_into_session(item):
    """
    Authoritatively loads an analysis history record into active session state.
    Ensures complete synchronization of results, filenames, titles, and authoritative score.
    """
    if not item:
        return
    res = {}
    if item.get("results_json"):
        try:
            res = json.loads(item["results_json"])
        except Exception:
            res = {}
            
    authoritative_score = get_overall_score(item.get("score") if item.get("score") is not None else res)
    if authoritative_score == 0.0 and res:
        authoritative_score = get_overall_score(res)
        
    if not res:
        res = {
            "scores": {
                "Language Quality": authoritative_score,
                "Structural Coherence": authoritative_score,
                "Argumentation": authoritative_score,
                "Academic Style": authoritative_score,
                "Readability": authoritative_score,
                "Composite": authoritative_score,
                "overall_score": authoritative_score
            },
            "domain": item.get("domain", "General Academic"),
            "full_text": ""
        }
        
    # Ensure scores dict exists and has authoritative values
    if "scores" not in res:
        res["scores"] = {}
    res["scores"]["Composite"] = authoritative_score
    res["scores"]["overall_score"] = authoritative_score
    res["overall_score"] = authoritative_score
    
    authoritative_filename = item.get("original_filename") or item.get("filename") or "Paper.pdf"
    clean_base = get_clean_base_filename(authoritative_filename)
    display_title = item.get("display_title") or get_clean_display_title(
        clean_base,
        text=res.get("full_text"),
        stored_title=res.get("title")
    )
    res["title"] = display_title
    res["display_title"] = display_title
    
    # Fully update session state
    st.session_state["results"] = res
    st.session_state["active_analysis"] = res
    st.session_state["analysis_result"] = res
    st.session_state["filename"] = authoritative_filename
    st.session_state["paper_title"] = display_title
    st.session_state["overall_score"] = authoritative_score
    st.session_state["current_analysis_id"] = item.get("analysis_id") or str(item.get("id"))
    st.session_state["chat_history"] = []

def is_generated_report_text(sentence):
    """Detects whether a sentence originates from PaperIQ generated report metadata rather than original manuscript."""
    s_lower = sentence.lower()
    report_markers = [
        "academic writing tone", "evaluation - formality", "objectivity:", "scholarly hedging",
        "promotional stance", "author framing", "paperiq", "analytical scoring framework",
        "responsible ai notice", "language quality", "structural coherence",
        "composite writing score", "reading difficulty", "executive summary",
        "citations detected", "lexical composition", "sentences exceed"
    ]
    return any(marker in s_lower for marker in report_markers)

def clean_heading_spacing(title):
    t = str(title).strip()
    # Separate jammed all-caps words (e.g. EMBEDDINGGENERATIONANDDATASET)
    tokens = [
        'EMBEDDING', 'GENERATION', 'DATASET', 'METHODOLOGY', 'PROPOSED',
        'FRAMEWORK', 'EXPERIMENTAL', 'EXPERIMENT', 'EVALUATION', 'RESULTS',
        'DISCUSSION', 'CONCLUSION', 'RELATED', 'WORK', 'SYSTEM', 'MODEL',
        'ARCHITECTURE', 'ALGORITHM', 'ANALYSIS', 'AND', 'FOR', 'WITH', 'OF',
        'THE', 'IN', 'ON', 'TO'
    ]
    pattern = '|'.join(tokens)
    spaced = re.sub(pattern, lambda m: m.group(0) + ' ', t).strip()
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', spaced)
    spaced = re.sub(r'\s+', ' ', spaced).strip()
    return spaced.title() if spaced.isupper() else spaced

def extract_sections(text):
    lines = text.split("\n")
    sections = {}
    current_header = "Preamble"
    current_content = []
    
    standard_sections = [
        "ABSTRACT", "INTRODUCTION", "LITERATURE REVIEW", "RELATED WORK",
        "METHODOLOGY", "METHODS", "PROPOSED METHOD", "SYSTEM MODEL", "DATASET",
        "EXPERIMENTAL SETUP", "RESULTS", "EXPERIMENTS", "DISCUSSION",
        "LIMITATIONS", "CONCLUSION", "FUTURE WORK", "REFERENCES"
    ]
    
    sec_counter = 1
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
            
        is_header = False
        candidate = line_s
        
        # Check if line is metadata noise
        if is_journal_or_metadata_noise(line_s):
            current_content.append(line_s)
            continue
            
        # Check standard numbering: I., II., A., B., 1., 1.1, etc.
        num_match = re.match(r"^([IVXLCDM]+|[A-Z]|\d+(\.\d+)*)[\.\s]\s*([A-Za-z].*)$", line_s)
        if num_match and len(line_s) < 70:
            is_header = True
            candidate = line_s
        elif any(line_s.upper().startswith(sh) for sh in standard_sections) and len(line_s) < 60:
            is_header = True
            candidate = line_s
        elif line_s.isupper() and 3 < len(line_s) < 45 and not any(char in line_s for char in [";", "$", "="]):
            is_header = True
            candidate = line_s
            
        if is_header:
            cleaned_title = clean_heading_spacing(candidate)
            if is_journal_or_metadata_noise(cleaned_title) or len(cleaned_title) < 3:
                cleaned_title = f"Section {sec_counter}"
                sec_counter += 1
                
            if current_content:
                sections[current_header] = " ".join(current_content).strip()
            current_header = cleaned_title
            current_content = []
        else:
            current_content.append(line_s)
            
    if current_content:
        sections[current_header] = " ".join(current_content).strip()
    return sections

def summarize_text(text, num_sentences=3):
    if not text:
        return "No content to summarize."
    blob = TextBlob(text)
    sentences = blob.sentences
    if len(sentences) <= num_sentences:
        return text
    word_frequencies = {}
    stop_words = set([
        "the", "is", "in", "and", "to", "of", "a", "for", "on", "with",
        "as", "by", "at", "this", "that", "it", "from", "an", "be", "are", "was"
    ])
    for word in blob.words:
        word_l = word.lower()
        if word_l not in stop_words and word_l.isalpha():
            word_frequencies[word_l] = word_frequencies.get(word_l, 0) + 1
    if not word_frequencies:
        return text
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_scores = {}
    for sent in sentences:
        for word in sent.words:
            word_l = word.lower()
            if word_l in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word_l]
    top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return " ".join([str(s) for s in top_sentences])

def get_important_sentences(text, num_sentences=3):
    sentences = re.split(r"(?<=[.!?]) +", text)
    if len(sentences) <= num_sentences:
        return sentences
    word_freq = {}
    words = re.findall(r"\w+", text.lower())
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    sentence_scores = {}
    for sentence in sentences:
        for word in re.findall(r"\w+", sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
    important = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return important

# ----------------------------------------------------
# Document Scoring & Readability Engine
# ----------------------------------------------------
def count_syllables(word):
    word = word.lower()
    vowels = "aeiou"
    count = 0
    if not word:
        return 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count <= 0:
        count = 1
    return count

def calculate_academic_readability(text):
    """
    Computes academic readability normalized for scholarly literature.
    Maps raw Flesch index to a fair 0-100 scale and provides Estimated Reading Difficulty.
    Prevents rigorous peer-reviewed papers from receiving an unjustified 0/100.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = [w for w in text.split() if w.strip()]
    total_words = len(words)
    total_sentences = len(sentences)
    if total_sentences == 0 or total_words == 0:
        return 70.0, "Moderate Academic", "Sufficient prose for evaluation."
    
    syllables = sum(count_syllables(w) for w in words)
    asl = total_words / max(1, total_sentences)
    asw = syllables / max(1, total_words)
    raw_flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
    
    if raw_flesch >= 55:
        norm_score = min(100.0, 75.0 + (raw_flesch - 55) * 0.5)
        difficulty = "Accessible Academic"
        desc = "Clear thematic accessibility with accessible sentence syntax."
    elif raw_flesch >= 35:
        norm_score = 80.0 + ((raw_flesch - 35) / 20.0) * 15.0
        difficulty = "Standard Academic"
        desc = "Well-balanced scholarly prose with conventional sentence density."
    elif raw_flesch >= 20:
        norm_score = 65.0 + ((raw_flesch - 20) / 15.0) * 15.0
        difficulty = "Dense Academic"
        desc = "Dense syntactic construction typical of peer-reviewed journal papers."
    elif raw_flesch >= 0:
        norm_score = 50.0 + (raw_flesch / 20.0) * 15.0
        difficulty = "Highly Complex"
        desc = "Elaborate multi-clause propositions requiring concentrated reviewer attention."
    else:
        norm_score = max(35.0, 50.0 + (raw_flesch * 0.25))
        difficulty = "Very Dense / Convoluted"
        desc = "Extremely long compound clauses; recommend segmenting into concise statements."
        
    return round(float(norm_score), 2), difficulty, desc

def calculate_readability(text):
    """Backwards compatibility alias returning normalized readability score."""
    score, _, _ = calculate_academic_readability(text)
    return score

def get_grade(score):
    """Returns PaperIQ Evaluation Grade (A, B, C, D, E)."""
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "E"

def interpret_score(score, metric_name):
    """Provides validated academic rating and contextual interpretation based on real score."""
    if score >= 85:
        rating = "Excellent"
        color = "#10B981"  # Emerald
    elif score >= 75:
        rating = "Very Good"
        color = "#3B82F6"  # Blue
    elif score >= 65:
        rating = "Good"
        color = "#8B5CF6"  # Purple
    elif score >= 50:
        rating = "Fair"
        color = "#F59E0B"  # Amber
    else:
        rating = "Needs Improvement"
        color = "#EF4444"  # Red

    descriptions = {
        "Language Quality": {
            "Excellent": "Strong academic vocabulary, clear sentence structure, and confident tone.",
            "Very Good": "Fluent academic writing with sound grammar and balanced syntax.",
            "Good": "Generally clear language with minor phrasing inconsistencies.",
            "Fair": "Passable language quality; some informal phrases or syntax irregularities.",
            "Needs Improvement": "Syntax and vocabulary require revision for scholarly rigor."
        },
        "Structural Coherence": {
            "Excellent": "Seamless logical transitions between paragraphs and concepts.",
            "Very Good": "Consistent flow of ideas with effective use of discourse markers.",
            "Good": "Understandable narrative; a few abrupt section shifts.",
            "Fair": "Paragraph linkages are somewhat disjointed; needs better transitional flow.",
            "Needs Improvement": "Fragmented arguments; lacks overarching structural narrative."
        },
        "Argumentation": {
            "Excellent": "Rigorously backed propositions with robust evidence and causal explanations.",
            "Very Good": "Solid analytical deductions with clear supporting data.",
            "Good": "Valid arguments present, though some claims lack direct evidentiary support.",
            "Fair": "Causal links are loosely asserted; requires stronger empirical citations.",
            "Needs Improvement": "Unsupported claims; argumentative chain needs fundamental grounding."
        },
        "Academic Style": {
            "Excellent": "Rich lexical diversity with specialized scholarly terminology.",
            "Very Good": "Substantial technical depth and appropriate domain-specific nomenclature.",
            "Good": "Adequate vocabulary with moderate variation of conceptual terms.",
            "Fair": "Repetitive vocabulary; relies on generic rather than precise academic verbs.",
            "Needs Improvement": "Limited lexical variety; significant repetition of basic phrasing."
        },
        "Readability": {
            "Excellent": "Optimal readability for scholarly evaluation without undue density.",
            "Very Good": "Well-balanced sentence rhythm and clear thematic accessibility.",
            "Good": "Readable academic prose; occasional dense clauses.",
            "Fair": "Dense syntactic structures typical of specialized research.",
            "Needs Improvement": "Extremely convoluted phrasing; sentences exceed typical readability thresholds."
        }
    }
    # Map legacy aliases
    alias_map = {
        "Language": "Language Quality",
        "Coherence": "Structural Coherence",
        "Reasoning": "Argumentation",
        "Sophistication": "Academic Style"
    }
    canonical = alias_map.get(metric_name, metric_name)
    interp = descriptions.get(canonical, {}).get(rating, "Analytical metric calculated from document characteristics.")
    return rating, color, interp

def analyze_academic_writing_tone(text, sections):
    """
    Reworked Academic Writing Tone Analysis replacing generic positive/negative sentiment.
    Evaluates formality, objectivity, hedging, claim strength, promotional tone, and author framing.
    """
    text_lower = text.lower()
    words = re.findall(r"\b[a-zA-Z]+\b", text_lower)
    word_count = max(1, len(words))
    
    # 1. Formality (Contraction avoidance & formal diction)
    contractions = ["don't", "can't", "won't", "didn't", "it's", "they're", "we've", "isn't", "aren't"]
    contraction_count = sum(len(re.findall(r"\b" + re.escape(c) + r"\b", text_lower)) for c in contractions)
    formality_score = min(100.0, max(0.0, 100.0 - (contraction_count * 5)))
    formality_status = "Formal Academic" if contraction_count == 0 else ("Generally Formal" if contraction_count <= 2 else "Contains Colloquial Contractions")
    
    # 2. Objectivity (Absence of emotive / subjective adjectives)
    blob = TextBlob(text)
    subjectivity = round(float(blob.sentiment.subjectivity), 2)
    objectivity_score = round(max(0.0, min(100.0, (1.0 - subjectivity) * 100)), 1)
    if objectivity_score >= 70:
        objectivity_status = "High Objectivity (Empirically Centered)"
    elif objectivity_score >= 50:
        objectivity_status = "Balanced (Moderate Subjective Commentary)"
    else:
        objectivity_status = "Subjective / Interpretive Framing"
        
    # 3. Hedging (Cautious scholarly assertions / epistemic modality)
    hedging_terms = [
        "suggests", "indicates", "appears to", "may indicate", "potential",
        "plausible", "tentatively", "presumably", "partially", "likely",
        "might", "could", "to our knowledge"
    ]
    hedge_count = sum(len(re.findall(r"\b" + re.escape(h) + r"\b", text_lower)) for h in hedging_terms)
    hedge_density = round((hedge_count / word_count) * 1000, 2)
    if hedge_density >= 4.0:
        hedging_status = "Robust Scholarly Hedging (Cautious Claims)"
    elif hedge_density >= 1.5:
        hedging_status = "Balanced Hedging"
    else:
        hedging_status = "Sparse Hedging (Claims Stated Directly)"
        
    # 4. Claim Strength (Definitive vs balanced claims)
    strong_claim_terms = [
        "proves", "proven", "undeniable", "incontestable", "irrefutable",
        "absolute truth", "completely solves", "flawless", "perfect solution"
    ]
    strong_claim_count = sum(len(re.findall(r"\b" + re.escape(sc) + r"\b", text_lower)) for sc in strong_claim_terms)
    claim_strength_status = "Balanced Academic Assertions" if strong_claim_count == 0 else f"{strong_claim_count} Absolute Assertions Detected"
    
    # 5. Promotional / Hype Language
    promo_terms = [
        "revolutionary", "groundbreaking", "game-changing", "miraculous",
        "unprecedented leap", "unbelievable", "world-changing", "disruptive breakthrough"
    ]
    promo_count = sum(len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower)) for p in promo_terms)
    promo_status = "Neutral Academic Stance (No Marketing Hype)" if promo_count == 0 else f"Promotional Tone Detected ({promo_count} hype phrases)"
    
    # 6. First-Person Framing
    first_person_singular = len(re.findall(r"\b(i|my|me|mine)\b", text_lower))
    first_person_plural = len(re.findall(r"\b(we|our|us|ours)\b", text_lower))
    if first_person_singular > 0:
        fp_status = f"Personal Voice Detected ({first_person_singular} single, {first_person_plural} collective)"
    elif first_person_plural > 0:
        fp_status = f"Author-Collective Framing ('we'/'our' used {first_person_plural} times)"
    else:
        fp_status = "Third-Person Passive Voice (Impersonal Academic)"
        
    return {
        "formality_score": formality_score,
        "formality_status": formality_status,
        "contraction_count": contraction_count,
        "objectivity_score": objectivity_score,
        "objectivity_status": objectivity_status,
        "subjectivity_raw": subjectivity,
        "hedging_count": hedge_count,
        "hedging_density": hedge_density,
        "hedging_status": hedging_status,
        "claim_strength_status": claim_strength_status,
        "strong_claim_count": strong_claim_count,
        "promotional_count": promo_count,
        "promotional_status": promo_status,
        "first_person_singular": first_person_singular,
        "first_person_plural": first_person_plural,
        "first_person_status": fp_status
    }

def analyze_full_document(text):
    if not text or len(text.strip()) < 50:
        return None
        
    blob = TextBlob(text)
    sentences = blob.sentences
    words = blob.words
    word_count = len(words)
    sentence_count = len(sentences)
    if sentence_count == 0 or word_count < 15:
        return None
        
    avg_sentence_len = np.mean([len(s.words) for s in sentences])
    avg_word_len = np.mean([len(w) for w in words])
    
    sections_data = extract_sections(text)
    tone_data = analyze_academic_writing_tone(text, sections_data)
    
    citation_patterns = [r"\(\d{4}\)", r"\[\d+\]", r"\(\w+ et al\., \d{4}\)"]
    citation_count = sum(len(re.findall(p, text)) for p in citation_patterns)
    
    # 1. Language Quality (0-100)
    # Balanced evaluation: sentence length calibration, word length, contraction penalty, lexical diversity
    len_penalty = min(20.0, abs(float(avg_sentence_len) - 22.0) * 0.9)
    word_adj = min(15.0, max(-10.0, (float(avg_word_len) - 5.0) * 7.5))
    contractions_penalty = min(20.0, float(tone_data.get("contraction_count", 0)) * 3.5)
    ttr = len(set([w.lower() for w in words])) / max(1, word_count)
    ttr_adj = min(10.0, max(-5.0, (ttr - 0.35) * 25.0))
    language_score = min(98.0, max(30.0, 76.0 - len_penalty + word_adj - contractions_penalty + ttr_adj))
    
    # 2. Structural Coherence (0-100)
    # Evaluates discourse transitions per sentence and presence of canonical manuscript sections
    transitions = [
        "however", "therefore", "thus", "consequently", "furthermore",
        "meanwhile", "moreover", "in addition", "in contrast", "subsequently",
        "accordingly", "specifically", "conversely", "notably"
    ]
    transition_count = sum(text.lower().count(t) for t in transitions)
    trans_density = transition_count / max(1, sentence_count)
    density_score = min(40.0, trans_density * 160.0)
    canonical_sections = ["abstract", "introduction", "method", "result", "conclusion"]
    sec_matches = sum(1 for cs in canonical_sections if any(cs in s.lower() for s in sections_data.keys()))
    sec_bonus = min(24.0, sec_matches * 5.0)
    coherence_score = min(98.0, max(30.0, 36.0 + density_score + sec_bonus))
    
    # 3. Argumentation (0-100)
    # Causal and evidence marker density plus empirical citation grounding
    argumentation_keywords = [
        "because", "since", "implies", "due to", "as a result", "evidence",
        "demonstrates", "proves", "corroborates", "substantiates", "suggests",
        "indicates", "validates", "confirms", "illustrates"
    ]
    arg_count = sum(text.lower().count(k) for k in argumentation_keywords)
    arg_density = arg_count / max(1, sentence_count)
    arg_density_score = min(42.0, arg_density * 170.0)
    cit_bonus = min(25.0, float(citation_count) * 2.5)
    argumentation_score = min(98.0, max(30.0, 32.0 + arg_density_score + cit_bonus))
    
    # 4. Academic Style (0-100)
    # Complex word ratio, formal objectivity, and promotional hype avoidance
    complex_words = [w for w in words if len(w) > 6]
    comp_ratio = len(complex_words) / max(1, word_count)
    style_base = 45.0 + min(36.0, max(0.0, (comp_ratio - 0.18) * 150.0))
    obj_score = float(tone_data.get("objectivity_score", 70.0))
    obj_adj = min(12.0, max(-10.0, (obj_score - 60.0) * 0.3))
    promo_penalty = min(20.0, float(tone_data.get("promotional_count", 0)) * 4.0)
    style_score = min(98.0, max(30.0, style_base + obj_adj - promo_penalty))
    
    # 5. Readability (0-100 normalized)
    readability_score, reading_difficulty, reading_desc = calculate_academic_readability(text)
    
    # Transparent composite score: (Language Quality 25%) + (Coherence 25%) + (Argumentation 20%) + (Style 15%) + (Readability 15%)
    final_score = calculate_overall_score(
        language_score, coherence_score, argumentation_score, style_score, readability_score
    )
    
    stats = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(float(avg_sentence_len), 2),
        "avg_word_len": round(float(avg_word_len), 2),
        "vocab_diversity": round(len(set([w.lower() for w in words])) / word_count, 2) if word_count else 0,
        "complex_word_ratio": round(len(complex_words) / word_count, 2) if word_count else 0,
        "reading_difficulty": reading_difficulty,
        "reading_difficulty_desc": reading_desc,
    }
    
    return {
        "scores": {
            "Language Quality": round(float(language_score), 2),
            "Structural Coherence": round(float(coherence_score), 2),
            "Argumentation": round(float(argumentation_score), 2),
            "Academic Style": round(float(style_score), 2),
            "Readability": round(float(readability_score), 2),
            "Composite": round(float(final_score), 1),
            "overall_score": round(float(final_score), 1),
            # Backwards-compatibility aliases
            "Language": round(float(language_score), 2),
            "Coherence": round(float(coherence_score), 2),
            "Reasoning": round(float(argumentation_score), 2),
            "Sophistication": round(float(style_score), 2),
        },
        "overall_score": round(float(final_score), 1),
        "stats": stats,
        "tone": tone_data,
        "sentiment": round(float(blob.sentiment.polarity), 2),
        "subjectivity": round(float(blob.sentiment.subjectivity), 2),
        "sections": sections_data,
    }

# ----------------------------------------------------
# Advanced NLP Insights & Suggestion Engines
# ----------------------------------------------------
ACADEMIC_VOCAB_SUGGESTIONS = [
    ("show", "demonstrate / exhibit", "Conveys rigorous empirical evidence rather than casual observation."),
    ("use", "utilize / employ", "More precise terminology in methodology and experimental design."),
    ("get", "obtain / derive", "Appropriate scholarly phrasing for results, data, and outcomes."),
    ("big", "substantial / significant", "Quantifies magnitude with academic specificity."),
    ("make", "formulate / construct", "Clearer action verb for models, theories, and protocols."),
    ("find", "identify / determine", "Reflects systematic investigation rather than incidental discovery."),
    ("good", "favorable / effective", "Specifies evaluation criteria rather than generic subjective value."),
    ("bad", "suboptimal / adverse", "Maintains objective, neutral academic tone in critique."),
    ("look at", "examine / investigate", "Standard terminology for literature reviews and methodology."),
    ("give", "provide / furnish", "Formal phrasing for presenting evidence or resources."),
    ("a lot of", "numerous / considerable", "Replaces informal quantifiers with precise scholarly vocabulary."),
    ("start", "initiate / commence", "Professional phrasing for workflows and experimental phases."),
    ("end", "conclude / terminate", "Conventional academic phrasing for processes and experiments."),
    ("tell", "indicate / signify", "More accurate description of data implications."),
]

def extract_vocabulary_improvements(text):
    text_lower = text.lower()
    results = []
    for original, suggestion, reason in ACADEMIC_VOCAB_SUGGESTIONS:
        matches = list(re.finditer(r"\b" + re.escape(original) + r"\b", text_lower))
        if matches:
            results.append({
                "original": original,
                "suggestion": suggestion,
                "reason": reason,
                "occurrences": len(matches)
            })
    return results

def detect_paper_issues(text, sections, scores, stats, citations):
    issues = []
    # 1. Structure: Missing Methodology
    # Only claim if section parsing is reliable (found at least 3 recognizable sections)
    if sections and len(sections) >= 3:
        has_methodology = any("method" in s.lower() for s in sections.keys())
        if not has_methodology:
            issues.append({
                "severity": "🔴 High",
                "category": "Structure",
                "problem": "Missing or unlabelled Methodology section",
                "issue": "Missing or unlabelled Methodology section",
                "why_it_matters": "Peer reviewers and readers need to understand the empirical procedure, dataset, or theoretical framework used in the research.",
                "recommendation": "Add a dedicated Methodology section describing the research procedure, dataset, tools, and evaluation process."
            })
            
    # 2. Citations
    total_cits = citations.get("total_citations", 0)
    has_ref_keyword = any(k in text.lower() for k in ["references", "bibliography", "works cited"])
    if total_cits == 0:
        if not has_ref_keyword:
            issues.append({
                "severity": "🟠 High",
                "category": "Citations",
                "problem": "No in-text citations or references section detected",
                "issue": "No in-text citations or references section detected",
                "why_it_matters": "Scholarly research requires rigorous attribution to situate claims within the broader academic corpus.",
                "recommendation": "Incorporate standard academic citations (e.g., [1] or Author, Year) and append a complete References section."
            })
        else:
            issues.append({
                "severity": "🟡 Moderate",
                "category": "Citations",
                "problem": "In-text citation markers could not be reliably extracted",
                "issue": "In-text citation markers could not be reliably extracted",
                "why_it_matters": "References heading was detected, but standard bracket or parenthetical markers were not located in the body prose.",
                "recommendation": "Ensure citation callouts adhere to standard conventions such as IEEE [1] or APA (Smith et al., 2023)."
            })
    elif total_cits < 5:
        issues.append({
            "severity": "🟡 Moderate",
            "category": "Citations",
            "problem": f"Sparse reference citations ({total_cits} citations detected)",
            "issue": f"Sparse reference citations ({total_cits} citations detected)",
            "why_it_matters": "A low citation count suggests related work and comparative baselines may be insufficiently contextualized.",
            "recommendation": "Incorporate authoritative peer-reviewed references to contextualize your findings within existing literature."
        })
        
    # 3. Readability: Excessively long sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    long_sentences = [s for s in sentences if len(s.split()) > 35 and not is_generated_report_text(s)]
    if len(long_sentences) >= 3:
        issues.append({
            "severity": "🟡 Medium",
            "category": "Readability",
            "problem": f"{len(long_sentences)} sentences exceed 35 words",
            "issue": f"{len(long_sentences)} sentences exceed 35 words",
            "why_it_matters": "Convoluted sentences increase cognitive load and obscure logical relationships between assertions.",
            "recommendation": "Segment convoluted sentences into concise, logical propositions to reduce cognitive load on reviewers."
        })
        
    # 4. Structural Coherence: Transition word density
    transitions = ["however", "therefore", "thus", "consequently", "furthermore", "meanwhile", "moreover", "in addition"]
    trans_count = sum(text.lower().count(t) for t in transitions)
    sentence_count = stats.get("sentence_count", 1)
    if sentence_count > 10 and (trans_count / max(1, sentence_count)) < 0.08:
        issues.append({
            "severity": "🟡 Medium",
            "category": "Structural Coherence",
            "problem": "Low density of transitional discourse markers",
            "issue": "Low density of transitional discourse markers",
            "why_it_matters": "Discourse markers help readers navigate logical relationships and transitions between consecutive paragraphs.",
            "recommendation": "Use transition words (e.g., 'Consequently', 'Furthermore', 'In contrast') to guide logical transitions between paragraphs."
        })
        
    # 5. Academic Style: Lexical diversity
    vocab_div = stats.get("vocab_diversity", 0.5)
    word_count = stats.get("word_count", 0)
    if vocab_div < 0.22 and word_count > 400:
        issues.append({
            "severity": "🔵 Low",
            "category": "Academic Style",
            "problem": "Low vocabulary diversity ratio",
            "issue": "Low vocabulary diversity ratio",
            "why_it_matters": "Repetitive non-technical phrasing diminishes scholarly rigor and precision.",
            "recommendation": "Expand scholarly terminology and avoid repeating the same non-technical words across consecutive sections."
        })
    return issues

def compute_section_sentiments(sections):
    results = []
    for title, content in sections.items():
        if content and len(content.strip()) > 30:
            blob = TextBlob(content)
            polarity = round(blob.sentiment.polarity, 3)
            subjectivity = round(blob.sentiment.subjectivity, 3)
            short_title = title if len(title) <= 22 else title[:19] + "..."
            if polarity > 0.05:
                tag = "Assertive 🟢"
                interp = "Optimistic / Assertive academic framing"
            elif polarity < -0.05:
                tag = "Critical 🔴"
                interp = "Critical / Problem-focused framing"
            else:
                tag = "Objective ⚪"
                interp = "Objective empirical reporting"
            subj_desc = "Objective" if subjectivity < 0.35 else ("Balanced" if subjectivity <= 0.6 else "Subjective")
            results.append({
                "full_title": title,
                "short_title": short_title,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment_tag": tag,
                "subjectivity_desc": subj_desc,
                "interpretation": interp
            })
    return results

def get_reliable_academic_sections(sections):
    """
    Filters detected sections strictly to recognized scholarly sections
    with sufficient textual content (at least 10 words).
    Returns dict of clean_section_name -> content.
    If fewer than 2 standard sections are found, returns empty dict to signal unreliable extraction.
    """
    if not sections or not isinstance(sections, dict):
        return {}
    canonical = [
        "Abstract", "Introduction", "Related Work", "Methodology", 
        "Results", "Discussion", "Conclusion", "References"
    ]
    matched = {}
    for canon in canonical:
        c_lower = canon.lower()
        for title, content in sections.items():
            t_lower = title.lower()
            is_match = False
            if c_lower in t_lower:
                is_match = True
            elif canon == "Methodology" and any(k in t_lower for k in ["method", "system model", "experimental design", "proposed model"]):
                is_match = True
            elif canon == "Related Work" and any(k in t_lower for k in ["literature review", "prior work", "background"]):
                is_match = True
            elif canon == "Results" and any(k in t_lower for k in ["findings", "experimental results", "evaluation"]):
                is_match = True
            elif canon == "Conclusion" and any(k in t_lower for k in ["concluding", "summary of work"]):
                is_match = True
                
            if is_match and len(content.strip().split()) >= 10:
                if canon not in matched:
                    matched[canon] = content.strip()
                    break
    if len(matched) < 2:
        return {}
    return matched

def extract_structured_research_gaps(text):
    """
    Extracts document-grounded research gaps, open problems, or author-stated limitations.
    Returns list of dicts:
      - evidence: exact excerpt from the text
      - limitation: concise statement of the specific limitation
      - why_gap: explanation of why this represents an open question
      - opportunity: concrete future research opportunity
      - confidence: 'High' or 'Moderate'
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    gap_markers = [
        ("limitation", "Author-stated study boundary or constraint", "Methodological boundaries restrict universal generalization", "High"),
        ("future work", "Author-identified open research inquiry", "Authors explicitly recommend further empirical validation", "High"),
        ("further research", "Unresolved investigation line", "Further experimental depth is warranted to confirm findings", "High"),
        ("however", "Contrasting constraint or caveat", "Contrasting observation indicates incomplete resolution of challenge", "Moderate"),
        ("challenging", "Analytical or computational bottleneck", "Technical challenge remains open for alternative techniques", "Moderate"),
        ("remains unclear", "Unresolved scientific inquiry", "Underlying mechanism is not definitively established in current literature", "High"),
        ("unresolved", "Open problem in empirical findings", "Evidence indicates an open challenge in the problem domain", "High"),
        ("drawback", "Identified architectural drawback", "Current implementation has known performance or scalability tradeoffs", "Moderate"),
        ("lack of", "Identified resource or data absence", "Absence of data or baseline hinders broader validation", "Moderate"),
        ("despite", "Persistent bottleneck", "Condition persists despite proposed intervention", "Moderate"),
    ]
    
    found_gaps = []
    seen_excerpts = set()
    
    for s in sentences:
        if is_generated_report_text(s):
            continue
        s_clean = re.sub(r"\s+", " ", s).strip()
        s_lower = s_clean.lower()
        if 40 < len(s_clean) < 250:
            for marker, lim_desc, gap_desc, conf in gap_markers:
                if marker in s_lower:
                    clean_excerpt = s_clean
                    if clean_excerpt not in seen_excerpts:
                        seen_excerpts.add(clean_excerpt)
                        opp = f"Conduct targeted empirical studies to overcome the '{marker}' constraint identified in the manuscript."
                        if "future work" in marker or "further research" in marker:
                            opp = "Directly pursue the future research extensions proposed by the manuscript authors."
                        elif "limitation" in marker:
                            opp = "Design expanded datasets or hybrid methodologies to mitigate this explicit limitation."
                            
                        found_gaps.append({
                            "evidence": clean_excerpt,
                            "limitation": lim_desc,
                            "why_gap": gap_desc,
                            "opportunity": opp,
                            "confidence": conf
                        })
                        break
        if len(found_gaps) >= 3:
            break
            
    return found_gaps

def extract_research_gaps(text):
    """Backwards compatibility alias returning string list."""
    structured = extract_structured_research_gaps(text)
    if not structured:
        return [
            "Not enough evidence in the document to determine specific open research gaps."
        ]
    return [g["evidence"] for g in structured]

def extract_key_contributions_structured(text):
    """
    Extracts author-stated contributions or inferences from empirical findings.
    Distinguishes author-stated vs inferred contributions without fabricating claims.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    explicit_markers = [
        "we propose", "in this paper", "this work presents", "our contribution",
        "we contribute", "main contribution", "key contribution", "primary contribution",
        "we demonstrate", "we develop", "we introduce", "first time", "our framework",
        "we design", "we present", "in this study, we"
    ]
    inferred_markers = [
        "results demonstrate", "achieves", "achieved", "outperforms", "outperformed",
        "findings indicate", "findings show", "results show", "results indicate",
        "significant improvement", "experiment shows", "experiments show", "observed that"
    ]
    
    contributions = []
    seen = set()
    
    for s in sentences:
        if is_generated_report_text(s):
            continue
        s_clean = re.sub(r"\s+", " ", s).strip()
        s_lower = s_clean.lower()
        if any(k in s_lower for k in explicit_markers) and 35 < len(s_clean) < 240:
            if s_clean not in seen:
                seen.add(s_clean)
                contributions.append({
                    "type": "Author-Stated Contribution",
                    "text": s_clean,
                    "confidence": "High"
                })
        if len(contributions) >= 3:
            break
            
    if len(contributions) < 2:
        for s in sentences:
            if is_generated_report_text(s):
                continue
            s_clean = re.sub(r"\s+", " ", s).strip()
            s_lower = s_clean.lower()
            if any(k in s_lower for k in inferred_markers) and 35 < len(s_clean) < 240:
                if s_clean not in seen:
                    seen.add(s_clean)
                    contributions.append({
                        "type": "Inferred from Empirical Findings",
                        "text": s_clean,
                        "confidence": "Moderate"
                    })
            if len(contributions) >= 3:
                break
                
    return contributions

def extract_key_contributions(text):
    """Backwards compatibility alias returning string list."""
    structured = extract_key_contributions_structured(text)
    if not structured:
        return [
            "Not enough evidence in the document to extract explicit author contribution claims."
        ]
    return [c["text"] for c in structured]

def extract_overly_long_sentences(text, max_words=35, limit=2):
    """Identifies concrete overly long sentences that degrade syntactic readability."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    long_ones = []
    for s in sentences:
        if is_generated_report_text(s):
            continue
        s_clean = re.sub(r"\s+", " ", s).strip()
        words = s_clean.split()
        if len(words) >= max_words and not any(ch in s_clean for ch in ["http", "www", "doi"]):
            long_ones.append((s_clean, len(words)))
            if len(long_ones) >= limit:
                break
    return long_ones

def extract_future_directions(text):
    """Extracts explicit future research directions or forward-looking statements from the manuscript."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    future_keywords = [
        "future work", "future research", "future investigations", "in the future",
        "next step", "promising avenue", "remains for future", "plan to explore",
        "further study", "extensions of this work", "will investigate", "future studies",
        "open direction"
    ]
    found = []
    for s in sentences:
        if is_generated_report_text(s):
            continue
        s_clean = re.sub(r"\s+", " ", s).strip()
        s_lower = s_clean.lower()
        if any(k in s_lower for k in future_keywords) and 30 < len(s_clean) < 250:
            found.append(s_clean)
            if len(found) >= 3:
                break
    if not found:
        return [
            "Not enough evidence in the document to extract explicit future research directions."
        ]
    return found

def extract_methodology_and_findings(text, sections=None):
    """Extracts grounded methodology summary, empirical findings, and data/materials from the manuscript."""
    meth_summary = ""
    findings_summary = ""
    data_summary = ""
    
    # Check structured sections first if available
    if sections and isinstance(sections, dict):
        for sec_name, sec_text in sections.items():
            sec_lower = sec_name.lower()
            if any(k in sec_lower for k in ["method", "approach", "framework", "architecture", "experimental setup"]):
                if not meth_summary and len(sec_text.strip()) > 40:
                    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.!?]+", sec_text) if len(s.strip()) > 30]
                    meth_summary = " ".join(sents[:2]) if sents else re.sub(r"\s+", " ", sec_text[:200]).strip()
            if any(k in sec_lower for k in ["result", "finding", "experiment", "evaluation", "discussion"]):
                if not findings_summary and len(sec_text.strip()) > 40:
                    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.!?]+", sec_text) if len(s.strip()) > 30]
                    findings_summary = " ".join(sents[:2]) if sents else re.sub(r"\s+", " ", sec_text[:200]).strip()
            if any(k in sec_lower for k in ["dataset", "material", "cohort", "data collection"]):
                if not data_summary and len(sec_text.strip()) > 30:
                    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.!?]+", sec_text) if len(s.strip()) > 25]
                    data_summary = " ".join(sents[:2]) if sents else re.sub(r"\s+", " ", sec_text[:180]).strip()

    # Fallback to sentence search across text
    sentences = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not meth_summary:
        m_keys = ["we implement", "we evaluate", "experimental setup", "we trained", "proposed architecture", "methodology", "baseline"]
        for s in sentences:
            if is_generated_report_text(s):
                continue
            if any(k in s.lower() for k in m_keys) and 35 < len(s) < 220:
                meth_summary = s
                break
    if not findings_summary:
        f_keys = ["results indicate", "results show", "outperforms", "achieves", "our findings demonstrate", "the model achieved", "empirically", "accuracy"]
        for s in sentences:
            if is_generated_report_text(s):
                continue
            if any(k in s.lower() for k in f_keys) and 35 < len(s) < 220:
                findings_summary = s
                break
    if not data_summary:
        d_keys = ["dataset", "cohort", "sample of", "materials", "telemetry", "field trial", "cultivars", "benchmark data", "admissions", "participants"]
        for s in sentences:
            if is_generated_report_text(s):
                continue
            if any(k in s.lower() for k in d_keys) and 30 < len(s) < 200:
                data_summary = s
                break
                
    if not meth_summary:
        meth_summary = "Not explicitly reported in the analyzed text."
    if not findings_summary:
        findings_summary = "Not explicitly reported in the analyzed text."
    if not data_summary:
        data_summary = "Not explicitly reported in the analyzed text."
        
    return {"methodology": meth_summary, "findings": findings_summary, "data_and_materials": data_summary}

ACADEMIC_DOMAIN_TAXONOMY = {
    "Agricultural & Food Sciences": [
        "agriculture", "agricultural", "crop", "crops", "farming", "farm", "farms", "soil", "yield",
        "cultivar", "cultivars", "cultivation", "harvest", "livestock", "agronomy", "irrigation",
        "fertilizer", "fertilizers", "pest", "pesticide", "grain", "wheat", "rice", "maize",
        "horticulture", "forestry", "agroforestry", "plant pathology", "food production", "seed",
        "germination", "agrochemical", "fodder", "silage", "pasture", "canopy", "tillage",
        "biochar", "crop yield", "plant growth", "soil health", "food security"
    ],
    "Biological & Life Sciences": [
        "biology", "biological", "genetics", "genome", "genomic", "cellular", "organism", "organisms",
        "ecology", "biodiversity", "species", "molecular biology", "physiology", "ecosystem", "dna",
        "rna", "protein", "proteins", "evolution", "botany", "zoology", "microbiology",
        "photosynthesis", "enzyme", "enzymes", "bacterial", "membrane", "metabolic", "pathogen"
    ],
    "Medicine & Health Sciences": [
        "clinical", "patient", "patients", "hospital", "therapy", "therapeutic", "pathology", "medical",
        "diagnosis", "diagnostic", "treatment", "treatments", "surgery", "surgical", "oncology",
        "pharmacological", "pharmacology", "epidemiology", "pharmacy", "vaccine", "cardiovascular",
        "physician", "morbidity", "mortality rate", "biomedical", "disease prevention", "healthcare"
    ],
    "Computer Science & Artificial Intelligence": [
        "algorithm", "algorithms", "neural network", "machine learning", "deep learning",
        "artificial intelligence", "nlp", "computer vision", "transformer", "transformers",
        "embeddings", "classification model", "reinforcement learning", "supervised learning",
        "unsupervised learning", "convolutional", "backpropagation", "benchmark dataset",
        "large language model", "latent space", "semantic segmentation", "hyperparameter"
    ],
    "Cybersecurity & Cryptography": [
        "cybersecurity", "cryptography", "cryptographic", "cyber attack", "vulnerability",
        "vulnerabilities", "malware", "ransomware", "intrusion detection", "encryption",
        "decryption", "adversarial attack", "phishing", "firewall", "zero-day",
        "authentication protocol", "exploit", "threat actor", "security penetration"
    ],
    "Networks & Distributed Systems": [
        "distributed systems", "cloud computing", "peer-to-peer", "consensus protocol",
        "byzantine", "latency", "throughput", "routing protocol", "kubernetes",
        "fault tolerance", "cluster computing", "blockchain", "load balancing", "serverless",
        "microservices", "network topology", "distributed consensus"
    ],
    "Physical Sciences & Mathematics": [
        "physics", "quantum", "thermodynamics", "astrophysics", "electromagnetism", "optics",
        "relativity", "particle physics", "mathematics", "mathematical theorem", "topology",
        "calculus", "differential equation", "algebraic", "numerical analysis",
        "probability distribution", "gravitational", "condensed matter", "hamiltonian"
    ],
    "Chemical & Materials Sciences": [
        "chemistry", "chemical", "chemical synthesis", "polymer", "polymers", "catalyst",
        "catalysts", "reagent", "molecular structure", "spectroscopy", "nanoparticles",
        "nanomaterials", "electrochemistry", "crystallography", "metallurgy", "inorganic chemistry",
        "organic synthesis", "crystallographic", "chemical bond", "thermogravimetric"
    ],
    "Earth & Environmental Sciences": [
        "environmental science", "climate change", "hydrology", "geology", "geological",
        "meteorology", "atmospheric science", "carbon emissions", "oceanography",
        "environmental pollution", "ecosystem preservation", "geospatial", "remote sensing",
        "precipitation", "temperature anomalies", "greenhouse gas", "groundwater"
    ],
    "Engineering & Robotics": [
        "mechanical engineering", "robotics", "actuator", "actuators", "kinematics",
        "control systems", "robot", "electrical grid", "structural engineering",
        "signal processing", "telecommunications", "embedded systems", "sensors",
        "microcontroller", "finite element", "aerospace", "mechatronics"
    ],
    "Economics & Business": [
        "economics", "econometric", "macroeconomic", "microeconomic", "financial markets",
        "inflation", "gdp", "market volatility", "portfolio optimization", "asset pricing",
        "monetary policy", "fiscal policy", "trade volume", "consumer behavior", "interest rate"
    ],
    "Social Sciences & Humanities": [
        "sociology", "psychology", "pedagogy", "educational curriculum", "linguistics",
        "anthropology", "political science", "philosophy", "ethics", "qualitative research",
        "ethnography", "cognitive development", "historical analysis", "governance policy"
    ]
}

def classify_research_domain(text):
    """
    Rigorously classifies document research domain using frequency-weighted token modeling.
    Distinguishes primary domain, secondary candidates, and confidence levels.
    Prevents cross-domain pollution (e.g. agricultural papers mistakenly assigned cloud or cybersecurity).
    """
    if not text:
        return {
            "primary": "Domain Uncertain",
            "secondary": [],
            "all_candidates": ["Domain Uncertain"],
            "confidence": "Uncertain",
            "scores": {}
        }
        
    text_lower = text.lower()
    domain_scores = {}
    
    for domain, terms in ACADEMIC_DOMAIN_TAXONOMY.items():
        score = 0
        for term in terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                weight = 3 if " " in term else 1
                score += matches * weight
        if score > 0:
            domain_scores[domain] = score
            
    if not domain_scores:
        return {
            "primary": "General Academic",
            "secondary": [],
            "all_candidates": ["General Academic"],
            "confidence": "Uncertain",
            "scores": {}
        }
        
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = sorted_domains[0]
    second_domain, second_score = sorted_domains[1] if len(sorted_domains) > 1 else (None, 0)
    
    if top_score < 3:
        return {
            "primary": "Domain Uncertain",
            "secondary": [],
            "all_candidates": ["Domain Uncertain"],
            "confidence": "Uncertain",
            "scores": domain_scores
        }
        
    if top_score >= 8 and (second_score == 0 or (top_score / max(1, second_score)) >= 1.7):
        confidence = "High"
    elif top_score >= 4:
        confidence = "Moderate"
    else:
        confidence = "Uncertain"
        
    secondary = []
    for d, s in sorted_domains[1:]:
        if s >= 4 and s >= 0.40 * top_score and len(secondary) < 2:
            secondary.append(d)
            
    all_candidates = [top_domain] + secondary
    
    return {
        "primary": top_domain,
        "secondary": secondary,
        "all_candidates": all_candidates,
        "confidence": confidence,
        "scores": domain_scores
    }

def extract_potential_research_areas(text):
    """Backwards compatibility wrapper returning candidate research domains."""
    clf = classify_research_domain(text)
    return clf["all_candidates"]

def extract_keywords_and_domain(text, top_n=12):
    blob = TextBlob(text)
    stop_words = set([
        "the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "as",
        "by", "at", "this", "that", "it", "from", "an", "be", "are", "was",
        "which", "were", "been", "have", "has", "using", "used", "paper"
    ])
    word_freq = {}
    for word in blob.words:
        word_l = word.lower()
        if word_l.isalpha() and word_l not in stop_words and len(word_l) > 3:
            word_freq[word_l] = word_freq.get(word_l, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in sorted_words[:top_n]]
    
    clf = classify_research_domain(text)
    return keywords, clf["all_candidates"]

def detect_publisher(text):
    publishers = {
        "IEEE": ["ieee", "ieeexplore"],
        "Springer Nature": ["springer", "springerlink", "nature"],
        "Elsevier / ScienceDirect": ["elsevier", "sciencedirect"],
        "Wiley Online Library": ["wiley"],
        "Taylor & Francis": ["taylor", "francis"],
        "ACM": ["acm", "association for computing machinery"],
        "MDPI": ["mdpi"],
        "Oxford University Press": ["oxford university press"],
    }
    text_lower = text.lower()
    for publisher, keywords in publishers.items():
        for word in keywords:
            if word in text_lower:
                return publisher
    return "Independent Academic Manuscript / Pre-print"

def analyze_citations(text):
    citation_patterns = [r"\(\d{4}\)", r"\[\d+\]", r"\(\w+ et al\., \d{4}\)"]
    citation_count = sum(len(re.findall(p, text)) for p in citation_patterns)
    total_words = len(text.split())
    citation_density = round(citation_count / total_words, 4) if total_words > 0 else 0.0
    impact_score = min(100.0, citation_count * 2.5)
    years = [int(y) for y in re.findall(r"(19\d{2}|20\d{2})", text) if 1970 <= int(y) <= 2026]
    avg_year = round(sum(years) / len(years), 1) if years else 0.0
    return {
        "total_citations": citation_count,
        "citation_density": citation_density,
        "impact_score": round(impact_score, 1),
        "average_year": avg_year,
    }

def calculate_semantic_strength(text):
    try:
        blob = TextBlob(text)
        nouns = [word.lower() for word, tag in blob.tags if tag.startswith("NN")]
        unique_nouns = len(set(nouns))
        total_nouns = len(nouns)
        if total_nouns == 0:
            return 0.0
        return round(min(100.0, (unique_nouns / total_nouns) * 150), 2)
    except Exception:
        words = text.split()
        return round(min(100.0, (len(set(words)) / max(1, len(words))) * 100), 2)

def novelty_score(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return round((len(set(words)) / len(words)) * 100, 2)

def generate_research_feedback(results):
    feedback = []
    scores = results.get("scores", {})
    comp = scores.get("Composite", 0)
    citations = results.get("citation_analysis", {}).get("total_citations", 0)
    sections = results.get("sections", {})
    if comp < 70:
        feedback.append("Overall academic writing and structure require targeted revision.")
    if citations < 5:
        feedback.append("Incorporate more scholarly citations to anchor empirical findings.")
    has_methodology = any("method" in k.lower() for k in sections.keys())
    if not has_methodology:
        feedback.append("Methodology section is absent or insufficiently labeled.")
    if not feedback:
        feedback.append("Paper demonstrates balanced academic structure and sound prose quality.")
    return feedback

def recommend_journal(domain):
    """Backwards-compatibility alias returning inferred research areas."""
    if isinstance(domain, list) and domain:
        return ", ".join(domain[:2])
    return str(domain)

def grounded_paper_qa(question, full_text, sections):
    """
    Answers questions strictly grounded in the analyzed document.
    Never hallucinates or uses external facts.
    Returns (answer_string, list_of_grounded_section_sources).
    """
    if not question or not full_text:
        return "Please ask a question regarding the uploaded document.", []
        
    question_clean = question.strip()
    question_lower = question_clean.lower()
    
    section_targets = {
        "objective": ["Abstract", "Introduction"],
        "goal": ["Abstract", "Introduction"],
        "aim": ["Abstract", "Introduction"],
        "problem": ["Introduction", "Abstract"],
        "contribution": ["Introduction", "Conclusion", "Abstract"],
        "methodology": ["Methodology", "Methods", "Proposed Method", "System Model"],
        "method": ["Methodology", "Methods", "Proposed Method"],
        "algorithm": ["Methodology", "Proposed Method", "Experimental Setup"],
        "dataset": ["Dataset", "Experimental Setup", "Methodology"],
        "data": ["Dataset", "Experimental Setup", "Methodology"],
        "model": ["Methodology", "System Model", "Proposed Method"],
        "finding": ["Results", "Discussion", "Conclusion"],
        "result": ["Results", "Experiments", "Discussion"],
        "metric": ["Experimental Setup", "Results", "Evaluation"],
        "evaluation": ["Experimental Setup", "Results"],
        "limitation": ["Limitations", "Discussion", "Conclusion"],
        "gap": ["Introduction", "Related Work"],
        "future work": ["Conclusion", "Future Work", "Discussion"],
        "conclusion": ["Conclusion", "Summary"]
    }
    
    matched_sources = []
    target_sections_to_search = []
    
    for kw, sec_names in section_targets.items():
        if kw in question_lower:
            for s_title, s_content in sections.items():
                if any(sn.lower() in s_title.lower() for sn in sec_names):
                    if len(s_content.strip()) > 30 and s_title not in target_sections_to_search:
                        target_sections_to_search.append(s_title)
                        
    if target_sections_to_search:
        extracted_paras = []
        for s_title in target_sections_to_search[:2]:
            s_content = sections[s_title]
            summary = summarize_text(s_content, num_sentences=3)
            if summary and len(summary.strip()) > 30:
                extracted_paras.append(summary)
                matched_sources.append(f"Section: {s_title}")
        if extracted_paras:
            return " ".join(extracted_paras), matched_sources
            
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.strip()) > 25]
    if not sentences:
        return "I couldn't find enough evidence in the analyzed document to answer that confidently.", []
        
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences + [question_clean])
        vectors = tfidf.toarray()
        cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])[0]
        best_idx = int(np.argmax(cosine_sim))
        best_sim = float(cosine_sim[best_idx])
        
        if best_sim < 0.12:
            return "I couldn't find enough evidence in the analyzed document to answer that confidently.", []
            
        top_indices = [int(i) for i in np.argsort(cosine_sim)[::-1][:3] if cosine_sim[i] > 0.08]
        selected_sentences = [sentences[i] for i in top_indices]
        
        for s_title, s_content in sections.items():
            if sentences[best_idx] in s_content:
                matched_sources.append(f"Section: {s_title}")
                break
        if not matched_sources and sections:
            first_sec = list(sections.keys())[0]
            matched_sources.append(f"Section: {first_sec}")
            
        return " ".join(selected_sentences), matched_sources
    except Exception:
        return "I couldn't find enough evidence in the analyzed document to answer that confidently.", []

def semantic_answer(question, full_text, sections):
    """Backwards compatibility alias returning answer text only."""
    answer, _ = grounded_paper_qa(question, full_text, sections)
    return answer

# ----------------------------------------------------
# PDF Report Generator (Strict Pure Bytes Return)
# ----------------------------------------------------
def clean_for_pdf(text):
    if not text:
        return ""
    return str(text).encode("latin-1", "ignore").decode("latin-1")

def generate_pdf_report(res, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    
    clean_base = get_clean_base_filename(filename)
    clean_disp = get_clean_display_title(clean_base, text=res.get("full_text"), stored_title=res.get("title"))
    
    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "PAPERIQ RESEARCH ANALYSIS REPORT", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "Academic Writing & Structure Evaluation | AI Research Intelligence", ln=True, align="C")
    pdf.ln(5)
    
    # Document Meta
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "1. Document Metadata", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, clean_for_pdf(f"Document Name: {clean_disp}"), ln=True)
    pdf.cell(0, 6, clean_for_pdf(f"Original File: {filename}"), ln=True)
    pdf.cell(0, 6, clean_for_pdf(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)
    areas_list = res.get("domains") or res.get("potential_research_areas") or ["General Academic"]
    pdf.cell(0, 6, clean_for_pdf(f"Potential Research Areas: {', '.join(areas_list)}"), ln=True)
    pdf.cell(0, 6, clean_for_pdf(f"Publisher Index: {res.get('publisher', 'Peer-Reviewed Manuscript')}"), ln=True)
    pdf.ln(4)

    # Executive Document Summary (if generated)
    cur_sum_text = None
    if isinstance(res, dict) and res.get("summary"):
        cur_sum_text = res.get("summary")
    elif "summary_result" in st.session_state:
        s_obj = st.session_state.get("summary_result")
        if s_obj and s_obj.get("success"):
            cur_sum_text = s_obj.get("summary")
            
    if cur_sum_text:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Executive Document Summary", ln=True)
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 5, clean_for_pdf(cur_sum_text[:2500]))
        pdf.ln(4)
    
    # Scores
    scores = res.get("scores", {})
    comp = get_overall_score(res)
    grade = get_grade(comp)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, f"2. Academic Writing & Structure Score: {comp:.1f} / 100 (Grade {grade})", ln=True)
    pdf.set_font("Arial", size=10)
    
    dim_items = [
        ("Language Quality", scores.get("Language Quality", scores.get("Language", 0))),
        ("Structural Coherence", scores.get("Structural Coherence", scores.get("Coherence", 0))),
        ("Argumentation", scores.get("Argumentation", scores.get("Reasoning", 0))),
        ("Academic Style", scores.get("Academic Style", scores.get("Sophistication", 0))),
        ("Readability", scores.get("Readability", 0)),
    ]
    for dim_name, val in dim_items:
        rating, _, _ = interpret_score(val, dim_name)
        pdf.cell(0, 6, clean_for_pdf(f" - {dim_name}: {val:.1f} / 100 ({rating})"), ln=True)
    pdf.ln(4)
    
    # Academic Writing Tone
    tone = res.get("tone", {})
    if tone:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "3. Academic Writing Tone Evaluation", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, clean_for_pdf(f" - Formality: {tone.get('formality_status', 'Formal Academic')} (Score: {tone.get('formality_score', 100):.0f}/100)"), ln=True)
        pdf.cell(0, 6, clean_for_pdf(f" - Objectivity: {tone.get('objectivity_status', 'Balanced')} (Score: {tone.get('objectivity_score', 80):.0f}/100)"), ln=True)
        pdf.cell(0, 6, clean_for_pdf(f" - Scholarly Hedging: {tone.get('hedging_status', 'Balanced')} ({tone.get('hedging_count', 0)} markers)"), ln=True)
        pdf.cell(0, 6, clean_for_pdf(f" - Promotional Stance: {tone.get('promotional_status', 'Neutral Academic Stance')}"), ln=True)
        pdf.cell(0, 6, clean_for_pdf(f" - Author Framing: {tone.get('first_person_status', 'Author-Collective')}"), ln=True)
        pdf.ln(4)
        
    # Document Metrics
    stats = res.get("stats", {})
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "4. Corpus & Lexical Statistics", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f" - Total Word Count: {stats.get('word_count', 0):,}", ln=True)
    pdf.cell(0, 6, f" - Total Sentences: {stats.get('sentence_count', 0):,}", ln=True)
    pdf.cell(0, 6, f" - Avg Sentence Length: {stats.get('avg_sentence_len', 0):.2f} words", ln=True)
    pdf.cell(0, 6, clean_for_pdf(f" - Reading Difficulty: {stats.get('reading_difficulty', 'Standard Academic')}"), ln=True)
    pdf.ln(4)
    
    # Observations & Issues
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "5. Critical Observations & Writing Suggestions", ln=True)
    pdf.set_font("Arial", size=10)
    issues = res.get("issues", [])
    if issues:
        for iss in issues[:5]:
            pdf.multi_cell(0, 6, clean_for_pdf(f" - [{iss.get('category')}] {iss.get('issue')}: {iss.get('recommendation')}"))
    else:
        for point in res.get("ai_feedback", ["Manuscript exhibits sound academic structure."]):
            pdf.multi_cell(0, 6, clean_for_pdf(f" - {point}"))
    pdf.ln(4)
    
    # Section Summaries
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "6. Section-Wise Executive Summaries", ln=True)
    for title, content in res.get("sections", {}).items():
        if len(content.strip()) < 30:
            continue
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 6, clean_for_pdf(title))
        pdf.set_font("Arial", size=9)
        summary = summarize_text(content, 2)
        pdf.multi_cell(0, 5, clean_for_pdf(summary[:900]))
        pdf.ln(2)
        
    # Responsible AI Notice
    pdf.ln(4)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(0, 4, clean_for_pdf(
        "Responsible AI Notice: PaperIQ evaluates textual, linguistic, and structural characteristics. "
        "It does not independently verify scientific correctness, novelty, experimental validity, plagiarism, or journal acceptance."
    ))
    
    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1", errors="replace")
    elif isinstance(pdf_output, (bytes, bytearray)):
        return bytes(pdf_output)
    return bytes(pdf_output)

# ----------------------------------------------------
# UI Theme Architecture: Light, Dark & System Modes
# ----------------------------------------------------
def get_system_theme():
    """Detects system appearance preference (Windows registry, environment, or default Dark)."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return "Light" if val == 1 else "Dark"
    except Exception:
        return "Dark"

def get_effective_theme(theme_pref=None):
    """Resolves 'System' to current system mode, or returns explicit 'Light'/'Dark'."""
    if theme_pref is None:
        theme_pref = st.session_state.get("theme_preference", "System")
    if theme_pref == "System":
        return get_system_theme()
    return theme_pref

def get_chart_theme(theme_pref=None):
    """Returns Plotly layout color palette customized for the active theme."""
    eff = get_effective_theme(theme_pref)
    if eff == "Light":
        return {
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#F8FAFC",
            "font_color": "#0F172A",
            "grid_color": "#E2E8F0",
            "line_color": "#CBD5E1",
            "tick_color": "#475569",
            "polar_bg": "#F8FAFC",
            "legend_bg": "rgba(255, 255, 255, 0.95)",
            "legend_border": "#CBD5E1",
            "radar_line": "#7C3AED",
            "radar_fill": "rgba(124, 58, 237, 0.22)"
        }
    else:
        return {
            "paper_bgcolor": "#111827",
            "plot_bgcolor": "#111827",
            "font_color": "#E2E8F0",
            "grid_color": "#1F2937",
            "line_color": "#374151",
            "tick_color": "#94A3B8",
            "polar_bg": "#111827",
            "legend_bg": "rgba(17, 24, 39, 0.9)",
            "legend_border": "#374151",
            "radar_line": "#A78BFA",
            "radar_fill": "rgba(139, 92, 246, 0.28)"
        }

def inject_theme_css(theme_pref="System"):
    """
    Injects comprehensive CSS rules tailored for Light, Dark, or System mode.
    Maintains clean layout, high contrast, readable cards, and professional purple branding.
    """
    eff = get_effective_theme(theme_pref)
    
    # Common layout styles
    base_css = """
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1250px !important;
    }
    """
    
    if eff == "Light":
        theme_css = """
        /* Main Light Background */
        .stApp {
            background-color: #F8FAFC !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            color: #0F172A !important;
        }

        /* Headings: Crisp Dark Slate */
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }

        /* Global Text & Labels */
        .stApp p, .stApp span, .stApp label, .stApp li {
            color: #334155 !important;
        }
        .stApp b, .stApp strong {
            color: #0F172A !important;
        }

        /* Sidebar: Clean White with Subtle Border */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            padding-top: 1.2rem !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #0F172A !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #475569 !important;
        }

        /* Sidebar Logo */
        .sidebar-logo-card {
            padding: 10px 4px 16px 4px;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 16px;
        }
        .sidebar-logo-title {
            font-size: 20px;
            font-weight: 800;
            background: linear-gradient(135deg, #7C3AED, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.05em;
            margin: 0;
        }
        .sidebar-logo-sub {
            font-size: 11px;
            color: #64748B !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 2px;
        }

        /* User Card in Sidebar */
        .sidebar-user-card {
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 16px;
        }
        .sidebar-user-role {
            display: inline-block;
            background: rgba(124, 58, 237, 0.12);
            color: #6D28D9 !important;
            border: 1px solid rgba(124, 58, 237, 0.25);
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 12px;
            margin-top: 4px;
        }

        /* Cards & Containers */
        .iq-card {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
        }
        .iq-card h3, .iq-card h4 {
            margin-top: 0 !important;
            color: #0F172A !important;
        }

        /* Metric Displays */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 16px 18px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #64748B !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }
        [data-testid="stMetricValue"] div {
            color: #0F172A !important;
            font-size: 28px !important;
            font-weight: 800 !important;
        }

        /* Score Cards */
        .score-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px;
            height: 100%;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        }
        .score-card-title {
            font-size: 12px;
            font-weight: 700;
            color: #64748B !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .score-card-val {
            font-size: 28px;
            font-weight: 800;
            color: #0F172A !important;
            margin: 4px 0;
        }
        .score-card-interp {
            font-size: 12px;
            color: #475569 !important;
            line-height: 1.4;
            margin-top: 6px;
        }

        /* Buttons */
        .stButton > button {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button * {
            color: #1E293B !important;
        }
        .stButton > button:hover {
            background-color: #F1F5F9 !important;
            border-color: #94A3B8 !important;
            color: #0F172A !important;
        }

        /* Primary Accent Buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3) !important;
        }
        .stButton > button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6D28D9, #4338CA) !important;
            box-shadow: 0 6px 18px rgba(124, 58, 237, 0.4) !important;
        }

        /* Sidebar Navigation Buttons */
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100% !important;
            text-align: left !important;
            justify-content: flex-start !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            color: #475569 !important;
            border-radius: 8px !important;
            padding: 0.55rem 0.85rem !important;
            margin-bottom: 3px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            box-shadow: none !important;
            display: flex !important;
            align-items: center !important;
            transition: all 0.15s ease-in-out !important;
        }
        section[data-testid="stSidebar"] div.stButton > button p,
        section[data-testid="stSidebar"] div.stButton > button span,
        section[data-testid="stSidebar"] div.stButton > button div {
            color: inherit !important;
            font-weight: inherit !important;
            text-align: left !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
            border-color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, rgba(124, 58, 237, 0.14) 0%, rgba(99, 102, 241, 0.08) 100%) !important;
            color: #6D28D9 !important;
            border: 1px solid #C4B5FD !important;
            border-left: 4px solid #7C3AED !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(124, 58, 237, 0.12) !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span {
            color: #6D28D9 !important;
            font-weight: 700 !important;
        }

        /* Download Buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #4F46E5, #2563EB) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
        }
        .stDownloadButton > button * {
            color: #FFFFFF !important;
        }

        /* File Uploader Container */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 18px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #CBD5E1 !important;
            border-radius: 10px !important;
            background-color: #F8FAFC !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #475569 !important;
        }

        /* Form Inputs */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border-color: #CBD5E1 !important;
            color: #0F172A !important;
        }
        div[data-baseweb="select"] * {
            color: #0F172A !important;
        }
        div[data-testid="stTextInput"] input {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            color: #0F172A !important;
            padding: 8px 12px !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #7C3AED !important;
            box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
        }

        div[data-testid="stRadio"] label, div[data-testid="stRadio"] p, div[data-testid="stRadio"] span {
            color: #1E293B !important;
        }

        /* Chat Styling */
        .chat-msg-user {
            background-color: #F1F5F9 !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            margin-bottom: 10px !important;
            color: #0F172A !important;
        }
        .chat-msg-user * {
            color: #0F172A !important;
        }
        .chat-msg-bot {
            background-color: #FFFFFF !important;
            border-left: 4px solid #7C3AED !important;
            border-top: 1px solid #E2E8F0 !important;
            border-right: 1px solid #E2E8F0 !important;
            border-bottom: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            margin-bottom: 14px !important;
            color: #1E293B !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        }
        .chat-msg-bot * {
            color: #1E293B !important;
        }
        .chat-msg-bot div[style*="color: #A78BFA"] {
            color: #6D28D9 !important;
        }
        .chat-msg-bot div[style*="color: #F8FAFC"] {
            color: #0F172A !important;
        }

        /* Native Containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: transparent !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
        }
        [data-testid="stExpander"] summary {
            color: #0F172A !important;
        }
        [data-testid="stExpander"] summary * {
            color: #0F172A !important;
        }

        /* Inline color overrides for light mode */
        .stApp [style*="color: #FFFFFF"], .stApp [style*="color:#FFFFFF"] { color: #0F172A !important; }
        .stApp [style*="color: #CBD5E1"], .stApp [style*="color:#CBD5E1"] { color: #334155 !important; }
        .stApp [style*="color: #94A3B8"], .stApp [style*="color:#94A3B8"] { color: #64748B !important; }
        .stApp [style*="background: #111827"], .stApp [style*="background-color: #111827"] { background: #FFFFFF !important; background-color: #FFFFFF !important; border-color: #E2E8F0 !important; }
        .stApp [style*="border-color: #1F2937"], .stApp [style*="border: 1px solid #1F2937"] { border-color: #E2E8F0 !important; }
        .stApp [style*="border-bottom: 1px solid #1F2937"] { border-bottom: 1px solid #E2E8F0 !important; }
        .stApp [style*="background: rgba(30, 41, 59, 0.45)"] { background: #F1F5F9 !important; border-color: #CBD5E1 !important; }
        .stApp [style*="background: rgba(139, 92, 246, 0.12)"] { background: rgba(124, 58, 237, 0.08) !important; border-color: rgba(124, 58, 237, 0.25) !important; }
        """
    else:
        # Dark Theme (Existing Deep Navy Academic SaaS Design)
        theme_css = """
        /* Main Dark Navy Background */
        .stApp {
            background-color: #0B0F19 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            color: #E2E8F0 !important;
        }

        /* Headings: Clean Crisp White */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }

        /* Global Text */
        .stApp p, .stApp span, .stApp label, .stApp li {
            color: #CBD5E1 !important;
        }

        /* Sidebar: Dark Navy-Blue */
        section[data-testid="stSidebar"] {
            background-color: #0D1117 !important;
            border-right: 1px solid #1F2937 !important;
            padding-top: 1.2rem !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #CBD5E1 !important;
        }

        /* Sidebar Logo Header */
        .sidebar-logo-card {
            padding: 10px 4px 16px 4px;
            border-bottom: 1px solid #1F2937;
            margin-bottom: 16px;
        }
        .sidebar-logo-title {
            font-size: 20px;
            font-weight: 800;
            background: linear-gradient(135deg, #A78BFA, #60A5FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.05em;
            margin: 0;
        }
        .sidebar-logo-sub {
            font-size: 11px;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 2px;
        }

        /* User Card in Sidebar */
        .sidebar-user-card {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 16px;
        }
        .sidebar-user-role {
            display: inline-block;
            background: rgba(139, 92, 246, 0.2);
            color: #C4B5FD !important;
            border: 1px solid rgba(139, 92, 246, 0.4);
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 12px;
            margin-top: 4px;
        }

        /* Professional Card Components */
        .iq-card {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        }
        .iq-card h3, .iq-card h4 {
            margin-top: 0 !important;
            color: #FFFFFF !important;
        }

        /* Metric Display Overrides */
        [data-testid="stMetric"] {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 12px !important;
            padding: 16px 18px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #94A3B8 !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }
        [data-testid="stMetricValue"] div {
            color: #FFFFFF !important;
            font-size: 28px !important;
            font-weight: 800 !important;
        }

        /* Score Card Component */
        .score-card {
            background-color: #111827;
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 18px;
            height: 100%;
        }
        .score-card-title {
            font-size: 12px;
            font-weight: 700;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .score-card-val {
            font-size: 28px;
            font-weight: 800;
            color: #FFFFFF !important;
            margin: 4px 0;
        }
        .score-card-interp {
            font-size: 12px;
            color: #94A3B8 !important;
            line-height: 1.4;
            margin-top: 6px;
        }

        /* Buttons */
        .stButton > button {
            background-color: #1E293B !important;
            color: #F1F5F9 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button * {
            color: #F1F5F9 !important;
        }
        .stButton > button:hover {
            background-color: #334155 !important;
            border-color: #64748B !important;
            color: #FFFFFF !important;
        }

        /* Primary Accent Buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35) !important;
        }
        .stButton > button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6D28D9, #4338CA) !important;
            box-shadow: 0 6px 18px rgba(124, 58, 237, 0.45) !important;
        }

        /* Sidebar Navigation Buttons */
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100% !important;
            text-align: left !important;
            justify-content: flex-start !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            color: #94A3B8 !important;
            border-radius: 8px !important;
            padding: 0.55rem 0.85rem !important;
            margin-bottom: 3px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            box-shadow: none !important;
            display: flex !important;
            align-items: center !important;
            transition: all 0.15s ease-in-out !important;
        }
        section[data-testid="stSidebar"] div.stButton > button p,
        section[data-testid="stSidebar"] div.stButton > button span,
        section[data-testid="stSidebar"] div.stButton > button div {
            color: inherit !important;
            font-weight: inherit !important;
            text-align: left !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #1F2937 !important;
            color: #F8FAFC !important;
            border-color: #374151 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, rgba(124, 58, 237, 0.25) 0%, rgba(79, 70, 229, 0.15) 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #7C3AED !important;
            border-left: 4px solid #8B5CF6 !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.2) !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        /* Download Buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #4F46E5, #2563EB) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        }
        .stDownloadButton > button * {
            color: #FFFFFF !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #4338CA, #1D4ED8) !important;
        }

        /* File Uploader Container */
        [data-testid="stFileUploader"] {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 12px !important;
            padding: 18px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #374151 !important;
            border-radius: 10px !important;
            background-color: #0B0F19 !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #CBD5E1 !important;
        }

        /* Inputs & Form Controls */
        div[data-baseweb="select"] > div {
            background-color: #111827 !important;
            border-color: #374151 !important;
            color: #FFFFFF !important;
        }
        div[data-baseweb="select"] * {
            color: #FFFFFF !important;
        }
        div[data-testid="stTextInput"] input {
            background-color: #111827 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            padding: 8px 12px !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #8B5CF6 !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
        }

        /* Chat Styling */
        .chat-msg-user {
            background-color: #1E293B;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
            color: #F8FAFC !important;
        }
        .chat-msg-bot {
            background-color: #111827;
            border-left: 4px solid #8B5CF6;
            border-top: 1px solid #1F2937;
            border-right: 1px solid #1F2937;
            border-bottom: 1px solid #1F2937;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 14px;
            color: #CBD5E1 !important;
        }

        /* Native Container Card with Border */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: transparent !important;
        }
        """

    full_css = f"<style>{base_css}\n{theme_css}</style>"
    st.markdown(full_css, unsafe_allow_html=True)

# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------
auth.create_users_table()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"
if "theme_preference" not in st.session_state:
    st.session_state.theme_preference = "System"
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "summary_length" not in st.session_state:
    st.session_state.summary_length = "Medium"
if "summary_doc_text" not in st.session_state:
    st.session_state.summary_doc_text = ""
if "summary_doc_name" not in st.session_state:
    st.session_state.summary_doc_name = ""

# Inject active theme stylesheet
inject_theme_css(st.session_state.theme_preference)

def navigate_to(page_name):
    st.session_state["current_page"] = page_name

# Reusable Plotly Layout Preset (Theme-Aware)
DARK_LAYOUT = get_chart_theme(st.session_state.theme_preference)

# ----------------------------------------------------
# Authentication View (Sign In & Create Account Redesign)
# ----------------------------------------------------
if not st.session_state.authenticated:
    eff_auth_theme = get_effective_theme(st.session_state.theme_preference)
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"
    if "auth_role_choice" not in st.session_state:
        st.session_state.auth_role_choice = "🎓 STUDENT LOGIN"
        
    # Read vector logo SVG and encode as Base64 to prevent markdown indentation code leaks
    logo_img_html = ""
    try:
        import base64
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "paperiq_logo.svg")
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                b64_logo = base64.b64encode(f.read()).decode("utf-8")
            logo_img_html = f'<img src="data:image/svg+xml;base64,{b64_logo}" width="76" height="60" alt="PaperIQ Logo" style="display: block; margin: 0 auto; filter: drop-shadow(0 4px 10px rgba(2, 132, 199, 0.22));" />'
    except Exception:
        logo_img_html = ""
    if not logo_img_html:
        logo_img_html = '<div style="font-size: 42px; text-align: center; margin-bottom: 6px;">📄</div>'

    # Background SVG (Base64 data URL)
    bg_svg_css = ""
    try:
        bg_fname = "login_bg.svg" if eff_auth_theme == "Light" else "login_bg_dark.svg"
        bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", bg_fname)
        if os.path.exists(bg_path):
            with open(bg_path, "rb") as f:
                b64_bg = base64.b64encode(f.read()).decode("utf-8")
            bg_svg_css = f"url('data:image/svg+xml;base64,{b64_bg}')"
    except Exception:
        bg_svg_css = ""

    # Scoped Auth CSS with strict overflow control, centered layout, and pixel alignment
    if eff_auth_theme == "Light":
        auth_theme_css = f"""
        <style>
        /* Strict global overflow prevention to eliminate horizontal scrollbar */
        html, body, [data-testid="stAppViewContainer"], .main {{
            overflow-x: hidden !important;
            max-width: 100vw !important;
            box-sizing: border-box !important;
        }}
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        div[data-testid="collapsedControl"] {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        /* Soft Aqua & Research Illustration Background */
        .stApp {{
            background: radial-gradient(circle at 50% 28%, #FFFFFF 0%, #EBF6F8 42%, #DBF1F5 75%, #CEEBF2 100%) {f", {bg_svg_css} center center / cover no-repeat fixed" if bg_svg_css else ""} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }}
        /* Centered Responsive Container */
        .block-container {{
            max-width: 440px !important;
            width: 100% !important;
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            margin: 0 auto !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
        }}
        /* White Authentication Card */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: #FFFFFF !important;
            border: 1px solid #D8E7EE !important;
            border-radius: 18px !important;
            padding: 24px 26px !important;
            box-shadow: 0 16px 40px rgba(14, 42, 71, 0.08), 0 2px 8px rgba(0, 0, 0, 0.03) !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }}
        /* Segmented Student / Teacher Selector Track */
        .st-key-auth_role_radio,
        div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]) {{
            width: 100% !important;
            max-width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 auto 14px auto !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stRadio"] {{
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            max-width: 100% !important;
            background: #EEF5F8 !important;
            border: 1px solid #DCE7EE !important;
            border-radius: 28px !important;
            padding: 3px !important;
            gap: 4px !important;
            box-sizing: border-box !important;
            align-items: stretch !important;
            justify-content: center !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] {{
            flex: 1 1 50% !important;
            width: 50% !important;
            max-width: 50% !important;
            min-width: 0 !important;
            min-height: 36px !important;
            height: 36px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            border-radius: 24px !important;
            padding: 0 8px !important;
            margin: 0 !important;
            cursor: pointer !important;
            border: 1.5px solid transparent !important;
            box-sizing: border-box !important;
            transition: all 0.2s ease !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
            display: none !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] p,
        div[data-testid="stRadio"] label[data-baseweb="radio"] span {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            color: #62758A !important;
            letter-spacing: 0.02em !important;
            margin: 0 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
            background: #FFFFFF !important;
            border: 1.5px solid #1A6ED8 !important;
            box-shadow: 0 2px 8px rgba(26, 110, 216, 0.2) !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {{
            color: #0E2A47 !important;
        }}
        /* Form Inputs */
        div[data-testid="stTextInput"] {{
            margin-bottom: 6px !important;
            width: 100% !important;
        }}
        div[data-testid="stTextInput"] input {{
            background: #FFFFFF !important;
            border: 1.5px solid #D6E1EA !important;
            border-radius: 9px !important;
            height: 42px !important;
            padding: 8px 12px !important;
            color: #172B45 !important;
            font-size: 13.5px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stTextInput"] input:focus {{
            border-color: #1A6ED8 !important;
            box-shadow: 0 0 0 3px rgba(26, 110, 216, 0.15) !important;
        }}
        div[data-testid="stTextInput"] label p {{
            color: #172B45 !important;
            font-weight: 600 !important;
            font-size: 12.5px !important;
            margin-bottom: 2px !important;
        }}
        div[data-baseweb="select"] > div {{
            border: 1.5px solid #D6E1EA !important;
            border-radius: 9px !important;
        }}
        /* Remember Me & Checkbox */
        div[data-testid="stCheckbox"] {{
            margin: 0 !important;
            padding-top: 2px !important;
        }}
        div[data-testid="stCheckbox"] label {{
            margin: 0 !important;
            align-items: center !important;
        }}
        div[data-testid="stCheckbox"] label p {{
            font-size: 12.5px !important;
            color: #62758A !important;
            margin: 0 !important;
            white-space: nowrap !important;
        }}
        /* Primary LOG IN / REGISTER Action Button */
        button[key="btn_signin"],
        button[key="btn_register"] {{
            background: linear-gradient(180deg, #1A6ED8 0%, #1157B8 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 9px !important;
            padding: 11px 16px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            letter-spacing: 0.04em !important;
            box-shadow: 0 4px 14px rgba(26, 110, 216, 0.35) !important;
            width: 100% !important;
            margin-top: 8px !important;
            margin-bottom: 4px !important;
            transition: all 0.2s ease !important;
        }}
        button[key="btn_signin"]:hover,
        button[key="btn_register"]:hover {{
            background: linear-gradient(180deg, #1660C0 0%, #0E4B9F 100%) !important;
            box-shadow: 0 6px 18px rgba(26, 110, 216, 0.45) !important;
            transform: translateY(-1px) !important;
        }}
        button[key="btn_signin"] p,
        button[key="btn_register"] p {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}
        /* Secondary Demo Button */
        button[key="btn_demo"] {{
            background: #FFFFFF !important;
            color: #172B45 !important;
            border: 1.5px solid #D6E1EA !important;
            border-radius: 9px !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03) !important;
            width: 100% !important;
            margin-top: 4px !important;
            transition: all 0.2s ease !important;
        }}
        button[key="btn_demo"]:hover {{
            background: #F4F8FA !important;
            border-color: #B8CEDC !important;
            color: #0E2A47 !important;
        }}
        button[key="btn_demo"] p {{
            color: #172B45 !important;
            font-weight: 600 !important;
        }}
        /* Create Account Switch Link */
        button[key="btn_switch_signup"],
        button[key="btn_switch_login"] {{
            background: transparent !important;
            border: none !important;
            color: #1A6ED8 !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            white-space: nowrap !important;
            box-shadow: none !important;
            padding: 0 !important;
            text-align: right !important;
            justify-content: flex-end !important;
            margin: 0 !important;
        }}
        button[key="btn_switch_signup"]:hover,
        button[key="btn_switch_login"]:hover {{
            color: #0E4B9F !important;
            text-decoration: underline !important;
            background: transparent !important;
        }}
        button[key="btn_switch_signup"] p,
        button[key="btn_switch_login"] p,
        button[key="btn_switch_signup"] span,
        button[key="btn_switch_login"] span {{
            white-space: nowrap !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            color: #1A6ED8 !important;
        }}
        </style>
        """
    else:
        auth_theme_css = f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], .main {{
            overflow-x: hidden !important;
            max-width: 100vw !important;
            box-sizing: border-box !important;
        }}
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        div[data-testid="collapsedControl"] {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        .stApp {{
            background: radial-gradient(circle at 50% 28%, #111D30 0%, #0B1424 50%, #060B14 100%) {f", {bg_svg_css} center center / cover no-repeat fixed" if bg_svg_css else ""} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }}
        .block-container {{
            max-width: 440px !important;
            width: 100% !important;
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            margin: 0 auto !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: #111E2E !important;
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
            border-radius: 18px !important;
            padding: 24px 26px !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45) !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }}
        /* Segmented Student / Teacher Selector Track */
        .st-key-auth_role_radio,
        div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]) {{
            width: 100% !important;
            max-width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 auto 14px auto !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stRadio"] {{
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            max-width: 100% !important;
            background: #0B1424 !important;
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 28px !important;
            padding: 3px !important;
            gap: 4px !important;
            box-sizing: border-box !important;
            align-items: stretch !important;
            justify-content: center !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] {{
            flex: 1 1 50% !important;
            width: 50% !important;
            max-width: 50% !important;
            min-width: 0 !important;
            min-height: 36px !important;
            height: 36px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            border-radius: 24px !important;
            padding: 0 8px !important;
            margin: 0 !important;
            cursor: pointer !important;
            border: 1.5px solid transparent !important;
            box-sizing: border-box !important;
            transition: all 0.2s ease !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
            display: none !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] p,
        div[data-testid="stRadio"] label[data-baseweb="radio"] span {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            color: #94A3B8 !important;
            letter-spacing: 0.02em !important;
            margin: 0 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
            background: #111E2E !important;
            border: 1.5px solid #38BDF8 !important;
            box-shadow: 0 2px 8px rgba(56, 189, 248, 0.25) !important;
        }}
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {{
            color: #FFFFFF !important;
        }}
        div[data-testid="stTextInput"] {{
            margin-bottom: 6px !important;
            width: 100% !important;
        }}
        div[data-testid="stTextInput"] input {{
            background: #0B1424 !important;
            border: 1.5px solid #1E2E42 !important;
            border-radius: 9px !important;
            height: 42px !important;
            padding: 8px 12px !important;
            color: #F8FAFC !important;
            font-size: 13.5px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stTextInput"] input:focus {{
            border-color: #38BDF8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
        }}
        div[data-testid="stTextInput"] label p {{
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 12.5px !important;
            margin-bottom: 2px !important;
        }}
        div[data-baseweb="select"] > div {{
            background: #0B1424 !important;
            border: 1.5px solid #1E2E42 !important;
            border-radius: 9px !important;
            color: #F8FAFC !important;
        }}
        div[data-testid="stCheckbox"] {{
            margin: 0 !important;
            padding-top: 2px !important;
        }}
        div[data-testid="stCheckbox"] label {{
            margin: 0 !important;
            align-items: center !important;
        }}
        div[data-testid="stCheckbox"] label p {{
            font-size: 12.5px !important;
            color: #94A3B8 !important;
            margin: 0 !important;
            white-space: nowrap !important;
        }}
        button[key="btn_signin"],
        button[key="btn_register"] {{
            background: linear-gradient(180deg, #1A6ED8 0%, #1157B8 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 9px !important;
            padding: 11px 16px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            letter-spacing: 0.04em !important;
            box-shadow: 0 4px 14px rgba(26, 110, 216, 0.35) !important;
            width: 100% !important;
            margin-top: 8px !important;
            margin-bottom: 4px !important;
            transition: all 0.2s ease !important;
        }}
        button[key="btn_signin"]:hover,
        button[key="btn_register"]:hover {{
            background: linear-gradient(180deg, #1660C0 0%, #0E4B9F 100%) !important;
            box-shadow: 0 6px 18px rgba(26, 110, 216, 0.45) !important;
        }}
        button[key="btn_signin"] p,
        button[key="btn_register"] p {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}
        button[key="btn_demo"] {{
            background: #0B1424 !important;
            color: #E2E8F0 !important;
            border: 1.5px solid #1E2E42 !important;
            border-radius: 9px !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            width: 100% !important;
            margin-top: 4px !important;
            transition: all 0.2s ease !important;
        }}
        button[key="btn_demo"]:hover {{
            background: #111E2E !important;
            border-color: #38BDF8 !important;
            color: #FFFFFF !important;
        }}
        button[key="btn_demo"] p {{
            color: #E2E8F0 !important;
        }}
        button[key="btn_switch_signup"],
        button[key="btn_switch_login"] {{
            background: transparent !important;
            border: none !important;
            color: #38BDF8 !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            white-space: nowrap !important;
            box-shadow: none !important;
            padding: 0 !important;
            text-align: right !important;
            justify-content: flex-end !important;
            margin: 0 !important;
        }}
        button[key="btn_switch_signup"]:hover,
        button[key="btn_switch_login"]:hover {{
            color: #7DD3FC !important;
            text-decoration: underline !important;
        }}
        button[key="btn_switch_signup"] p,
        button[key="btn_switch_login"] p,
        button[key="btn_switch_signup"] span,
        button[key="btn_switch_login"] span {{
            white-space: nowrap !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            color: #38BDF8 !important;
        }}
        </style>
        """
    st.markdown(auth_theme_css, unsafe_allow_html=True)

    # Centered PaperIQ Branding Header (Zero Raw SVG / Rendered via Clean Vector Img Tag)
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 18px;">
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 8px;">
                {logo_img_html}
            </div>
            <div style="font-size: 28px; font-weight: 800; letter-spacing: 0.16em; color: {'#0E2A47' if eff_auth_theme == 'Light' else '#FFFFFF'}; margin: 0; line-height: 1.1;">PAPERIQ</div>
            <div style="font-size: 13.5px; font-weight: 600; color: {'#2C4E72' if eff_auth_theme == 'Light' else '#94A3B8'}; letter-spacing: 0.03em; margin-top: 5px;">AI-Powered Research Insight Analyzer</div>
            <div style="font-size: 11.5px; font-style: italic; color: {'#5A7B9D' if eff_auth_theme == 'Light' else '#64748B'}; margin-top: 2px;">Analyze. Understand. Improve.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Centered Authentication Card
    with st.container(border=True):
        if st.session_state.auth_mode == "login":
            # 1. Login Role Selector (Side-by-side, 50/50 width, no wrapping)
            role_options = ["🎓 STUDENT LOGIN", "👩‍🏫 TEACHER LOGIN"]
            sel_role = st.radio(
                "Login Role Selection",
                options=role_options,
                index=0 if "STUDENT" in st.session_state.auth_role_choice else 1,
                horizontal=True,
                label_visibility="collapsed",
                key="auth_role_radio"
            )
            st.session_state.auth_role_choice = sel_role
            is_student = "STUDENT" in sel_role

            # 2. Username Input (Full Width)
            login_user_input = st.text_input(
                "Username" if is_student else "Teacher Username",
                placeholder="Enter your username or Student ID" if is_student else "Enter your faculty or researcher username",
                key="auth_login_user"
            )

            # 3. Password Input (Full Width with Native Built-In Visibility Eye Toggle)
            login_pass_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="auth_login_pass"
            )

            # 4. Remember Me & Forgot Password (Aligned on One Row)
            col_rem, col_forgot = st.columns([1.1, 1.1])
            with col_rem:
                st.checkbox("Remember Me", value=True, key="auth_remember_me")
            with col_forgot:
                st.markdown(
                    f"""
                    <div style="text-align: right; padding-top: 3px; white-space: nowrap;">
                        <a href="mailto:support@paperiq.edu?subject=PaperIQ%20Account%20Assistance" style="color: {'#1A6ED8' if eff_auth_theme == 'Light' else '#38BDF8'}; font-size: 12.5px; font-weight: 600; text-decoration: none;">Forgot Password?</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 5. Primary Sign In Action
            if st.button("LOG IN", type="primary", use_container_width=True, key="btn_signin"):
                if login_user_input and login_pass_input:
                    user = auth.login_user(login_user_input, login_pass_input)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_data = user
                        st.session_state.theme_preference = user.get("theme_preference", "System")
                        st.session_state.analysis_history[user["username"]] = auth.get_user_history(user["username"])
                        st.success("Signed in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please verify your credentials.")
                else:
                    st.warning("Please enter your username and password.")

            # 6. Secondary Demo Action (Explicit Student Demo Access)
            if st.button("🎓 Continue as Demo Student", use_container_width=True, key="btn_demo"):
                demo_username = "Student_Demo"
                demo_role = "Student (Demo Access)"
                auth.signup_user(demo_username, "demo123", demo_role)
                st.session_state.authenticated = True
                st.session_state.user_data = {"username": demo_username, "role": demo_role}
                st.session_state.theme_preference = auth.get_user_theme(demo_username) or "System"
                st.session_state.analysis_history[demo_username] = auth.get_user_history(demo_username)
                st.rerun()

            # 7. Footer Switch to Registration (No Awkward Line Breaks)
            st.markdown("<div style='height: 12px; border-top: 1px solid rgba(214, 225, 234, 0.6); margin-top: 16px; margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            col_f1, col_f2 = st.columns([1.1, 1.3])
            with col_f1:
                st.markdown(
                    f"""<div style="font-size: 12.5px; color: {'#62758A' if eff_auth_theme == 'Light' else '#94A3B8'}; line-height: 38px; white-space: nowrap;">Don't have an account?</div>""",
                    unsafe_allow_html=True
                )
            with col_f2:
                if st.button("Create Account →", key="btn_switch_signup", use_container_width=True):
                    st.session_state.auth_mode = "signup"
                    st.rerun()

        else:
            # CREATE ACCOUNT (REGISTRATION) VIEW
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 16px;">
                    <div style="font-size: 18px; font-weight: 700; color: {'#0E2A47' if eff_auth_theme == 'Light' else '#FFFFFF'};">Create Your Account</div>
                    <div style="font-size: 12.5px; color: {'#62758A' if eff_auth_theme == 'Light' else '#94A3B8'}; margin-top: 3px;">Join PaperIQ to analyze and evaluate research papers</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            signup_user_input = st.text_input("Username", placeholder="Choose a username", key="auth_reg_user")

            signup_pass_input = st.text_input(
                "Password",
                type="password",
                placeholder="Create a password (min 4 chars)",
                key="auth_reg_pass"
            )

            signup_pass_confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter password",
                key="auth_reg_confirm"
            )

            signup_role_input = st.selectbox("Academic Role", ["Student", "Faculty", "Researcher", "Evaluator"], key="auth_reg_role")

            if st.button("REGISTER ACCOUNT", type="primary", use_container_width=True, key="btn_register"):
                if not signup_user_input or not signup_pass_input:
                    st.warning("Please complete all required fields.")
                elif len(signup_pass_input) < 4:
                    st.warning("Password must be at least 4 characters.")
                elif signup_pass_input != signup_pass_confirm:
                    st.error("Passwords do not match.")
                else:
                    success, msg = auth.signup_user(signup_user_input, signup_pass_input, signup_role_input)
                    if success:
                        st.success("Account created successfully! Please sign in with your credentials.")
                        st.session_state["auth_mode"] = "login"
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("<div style='height: 12px; border-top: 1px solid rgba(214, 225, 234, 0.6); margin-top: 16px; margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([1.1, 1.3])
            with col_b1:
                st.markdown(
                    f"""<div style="font-size: 12.5px; color: {'#62758A' if eff_auth_theme == 'Light' else '#94A3B8'}; line-height: 38px; white-space: nowrap;">Already have an account?</div>""",
                    unsafe_allow_html=True
                )
            with col_b2:
                if st.button("← Back to Sign In", key="btn_switch_login", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()

    st.stop()

# ----------------------------------------------------
# Authenticated State & Sidebar Redesign
# ----------------------------------------------------
username = st.session_state.user_data.get("username", "Student")
role = st.session_state.user_data.get("role", "Student")

# Ensure user theme preference is loaded if user was already authenticated
if username and username != "Student" and st.session_state.get("theme_preference") in [None, ""]:
    st.session_state["theme_preference"] = auth.get_user_theme(username) or "System"

# Keep history synchronized
if username not in st.session_state.analysis_history:
    st.session_state.analysis_history[username] = auth.get_user_history(username)
user_history = st.session_state.analysis_history.get(username, [])

with st.sidebar:
    # Sidebar Logo Header
    st.markdown(
        """
        <div class="sidebar-logo-card">
            <div class="sidebar-logo-title">PAPERIQ</div>
            <div class="sidebar-logo-sub">AI Research Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # User Profile Card
    st.markdown(
        f"""
        <div class="sidebar-user-card">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Active Account</div>
            <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{username}</div>
            <span class="sidebar-user-role">🎓 {role}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div style='font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin: 12px 0 6px 4px; letter-spacing: 0.08em;'>Workspace</div>", unsafe_allow_html=True)
    
    workspace_nav = [
        ("🏠 Dashboard", "nav_dashboard"),
        ("📄 Analyze Paper", "nav_analyze"),
        ("🔄 Compare Papers", "nav_compare"),
    ]
    
    current_pg = st.session_state.get("current_page", "🏠 Dashboard")
    for label, key in workspace_nav:
        is_active = (current_pg == label)
        st.button(
            label,
            key=key,
            on_click=navigate_to,
            args=(label,),
            use_container_width=True,
            type="primary" if is_active else "secondary"
        )
    
    st.markdown("<div style='font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin: 14px 0 6px 4px; letter-spacing: 0.08em;'>Results</div>", unsafe_allow_html=True)
    
    results_nav = [
        ("📊 Analysis", "nav_analysis"),
        ("💡 Insights", "nav_insights"),
        ("💬 Ask PaperIQ", "nav_ask"),
        ("📚 History", "nav_history"),
    ]
    for label, key in results_nav:
        is_active = (current_pg == label or (label == "📚 History" and current_pg in ["📚 History", "🕘 History"]))
        st.button(
            label,
            key=key,
            on_click=navigate_to,
            args=(label,),
            use_container_width=True,
            type="primary" if is_active else "secondary"
        )
        
    st.markdown("<div style='font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin: 14px 0 6px 4px; letter-spacing: 0.08em;'>Account</div>", unsafe_allow_html=True)
    
    account_nav = [
        ("⚙ Settings", "nav_settings"),
        ("ℹ About", "nav_about"),
    ]
    for label, key in account_nav:
        is_active = (current_pg == label or (label == "⚙ Settings" and current_pg in ["⚙ Settings", "⚙️ Settings"]) or (label == "ℹ About" and current_pg in ["ℹ About", "ℹ️ About"]))
        st.button(
            label,
            key=key,
            on_click=navigate_to,
            args=(label,),
            use_container_width=True,
            type="primary" if is_active else "secondary"
        )
        
    st.markdown("<div style='height: 12px; border-top: 1px solid #1F2937; margin: 14px 0 8px 0;'></div>", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True, key="btn_logout"):
        saved_theme = st.session_state.get("theme_preference", "System")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state["theme_preference"] = saved_theme
        st.rerun()

# ----------------------------------------------------
# PAGE 1: DASHBOARD
# ----------------------------------------------------
if st.session_state.current_page == "🏠 Dashboard":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Research Workspace</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Analyze, evaluate, and understand research papers with AI-powered insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_dash_btn1, col_dash_btn2, _ = st.columns([1.5, 1.5, 4])
    with col_dash_btn1:
        st.button("+ Analyze New Paper", type="primary", use_container_width=True, key="dash_btn_analyze", on_click=navigate_to, args=("📄 Analyze Paper",))
    with col_dash_btn2:
        st.button("🔄 Compare Papers", use_container_width=True, key="dash_btn_comp", on_click=navigate_to, args=("🔄 Compare Papers",))
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dynamic Summary Cards
    total_analyzed = len(user_history)
    avg_score = round(sum(get_overall_score(h) for h in user_history) / total_analyzed, 1) if total_analyzed > 0 else 0.0
    best_score = round(max((get_overall_score(h) for h in user_history), default=0.0), 1)
    reports_gen = sum(1 for h in user_history if h.get("pdf"))
    
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Papers Analyzed", total_analyzed)
    d2.metric("Average Writing Score", f"{avg_score} / 100" if total_analyzed > 0 else "N/A")
    d3.metric("Best Writing Score", f"{best_score} / 100" if total_analyzed > 0 else "N/A")
    d4.metric("Reports Generated", reports_gen)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Recent Analyses")
    
    if not user_history:
        st.markdown(
            """
            <div class="iq-card" style="text-align: center; padding: 40px !important;">
                <div style="font-size: 32px; margin-bottom: 8px;">📄</div>
                <h4 style="color: #CBD5E1; margin: 0;">No Analysis History Yet</h4>
                <p style="color: #64748B; margin-top: 4px;">Upload your first research paper in <b>Analyze Paper</b> to view evaluation metrics and reports here.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for idx, item in enumerate(user_history[:5]):
            item_score = get_overall_score(item)
            grade = get_grade(item_score)
            res_meta = None
            if item.get("results_json"):
                try:
                    res_meta = json.loads(item["results_json"])
                except Exception:
                    pass
            stored_title = item.get("display_title") or (res_meta.get("title") if res_meta else None)
            clean_display_title = get_clean_display_title(item.get('filename'), stored_title=stored_title)
            truncated_title = clean_display_title if len(clean_display_title) <= 45 else f"{clean_display_title[:42]}..."
            clean_fn = get_clean_report_filename(item.get('filename'))
            raw_date = item.get("created_at", "")
            try:
                analyzed_date_str = datetime.strptime(raw_date[:10], "%Y-%m-%d").strftime("%d %b %Y")
            except Exception:
                analyzed_date_str = raw_date
            
            with st.container(border=True):
                col_info, col_score, col_act1, col_act2 = st.columns([4.2, 2.2, 1.8, 1.8])
                with col_info:
                    st.markdown(
                        f"""
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;" title="{clean_display_title}">📄 {truncated_title}</div>
                        <div style="color: #94A3B8; font-size: 12px; margin-top: 3px;">
                            Domain: <b style="color: #60A5FA;">{item.get('domain', 'General Academic')}</b> &nbsp;•&nbsp; Analyzed: {analyzed_date_str}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_score:
                    st.markdown(
                        f"""
                        <div style="font-size: 15px; font-weight: 800; color: #A78BFA; margin-top: 6px;">
                            Score: {item_score:.1f} / 100 &nbsp;
                            <span style="background: rgba(139, 92, 246, 0.2); color: #C4B5FD; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 8px;">Grade {grade}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_act1:
                    if st.button("View Analysis", key=f"dash_view_{item.get('id', idx)}", use_container_width=True):
                        load_analysis_into_session(item)
                        st.session_state.current_page = "📊 Analysis"
                        st.rerun()
                with col_act2:
                    if item.get("pdf"):
                        st.download_button(
                            "Download Report",
                            data=item["pdf"],
                            file_name=clean_fn,
                            mime="application/pdf",
                            key=f"dash_down_{item.get('id', idx)}",
                            use_container_width=True
                        )

# ----------------------------------------------------
# PAGE 2: ANALYZE PAPER
# ----------------------------------------------------
elif st.session_state.current_page == "📄 Analyze Paper":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Analyze Your Research Paper</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Upload a research document for academic evaluation across language, structure, argumentation, and readability.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.45); border: 1px dashed rgba(167, 139, 250, 0.35); border-radius: 12px; padding: 18px 22px 14px 22px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h3 style="margin: 0; font-size: 17px; color: #FFFFFF; font-weight: 600;">📄 Research Paper Upload</h3>
                <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 6px; padding: 3px 9px; font-size: 11px; color: #C4B5FD; font-weight: 600;">Max 30 MB • PDF, DOCX, TXT</span>
            </div>
            <p style="color: #94A3B8; font-size: 13px; margin: 0;">Drag and drop your manuscript below or browse files. In-memory processing ensures full privacy and no permanent external storage.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "Upload research document",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        file_bytes_raw = uploaded_file.getvalue()
        file_size_mb = len(file_bytes_raw) / (1024 * 1024)
        ext = uploaded_file.name.rsplit(".", 1)[-1].upper()
        preview_title = get_clean_display_title(uploaded_file.name)
        
        st.markdown(
            f"""
            <div class="iq-card" style="margin-top: 14px; margin-bottom: 14px; padding: 18px 22px !important; border-left: 4px solid #10B981;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 11px; font-weight: 700; color: #10B981; letter-spacing: 0.08em; text-transform: uppercase;">✓ FILE UPLOADED &amp; VALIDATED</span>
                    <span style="background: rgba(16, 185, 129, 0.15); color: #34D399; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.25);">Ready for analysis</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px;">
                    <div><span style="color: #94A3B8; font-size: 12px;">Document:</span><br><b style="color: #FFFFFF; font-size: 14px; word-break: break-word;">📄 {preview_title}</b></div>
                    <div><span style="color: #94A3B8; font-size: 12px;">Original file:</span><br><b style="color: #CBD5E1; font-size: 13px; word-break: break-all;">{uploaded_file.name}</b></div>
                    <div><span style="color: #94A3B8; font-size: 12px;">Format:</span><br><b style="color: #A78BFA; font-size: 14px;">{ext}</b></div>
                    <div><span style="color: #94A3B8; font-size: 12px;">Size:</span><br><b style="color: #FFFFFF; font-size: 14px;">{file_size_mb:.2f} MB</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Invalidate previous summary if new file selected
        if st.session_state.get("summary_doc_name") != uploaded_file.name:
            st.session_state.summary_result = None
            st.session_state.summary_doc_text = ""
            st.session_state.summary_doc_name = uploaded_file.name

        tab_summarize, tab_eval = st.tabs(["📝 Summarize Document", "🚀 Academic Quality Evaluation"])

        with tab_summarize:
            st.markdown(
                """
                <div style="margin-bottom: 12px; margin-top: 6px;">
                    <h3 style="margin: 0; font-size: 19px; color: #FFFFFF; font-weight: 700;">Executive Document Summarization</h3>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 13.5px;">Generate source-grounded academic summaries of the complete paper across Short, Medium, or Long detail levels.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Document Info Summary Row
            cached_text = st.session_state.get("summary_doc_text", "")
            extraction_error = None
            if not cached_text and hasattr(uploaded_file, "getvalue"):
                try:
                    pre_extracted = extract_text_from_file(uploaded_file)
                    if pre_extracted and pre_extracted.startswith("__ERROR_"):
                        extraction_error = pre_extracted
                    elif pre_extracted:
                        cached_text = clean_text(pre_extracted)
                        st.session_state.summary_doc_text = cached_text
                except Exception as ex:
                    extraction_error = f"__ERROR_EXCEPTION__: {ex}"
            
            doc_word_count = len(re.findall(r"\b\w+\b", cached_text)) if cached_text else 0
            if doc_word_count > 0:
                doc_page_count = max(1, math.ceil(doc_word_count / 450))
            elif file_size_mb > 0 and not extraction_error:
                doc_page_count = max(1, math.ceil(file_size_mb * 2.5))
            else:
                doc_page_count = 0
            
            if extraction_error == "__ERROR_SCANNED_PDF__":
                status_badge_text = "⚠️ Scanned / Image PDF"
                status_badge_color = "#EF4444"
                status_badge_bg = "rgba(239, 68, 68, 0.15)"
            elif extraction_error == "__ERROR_PASSWORD_PROTECTED__":
                status_badge_text = "⚠️ Encrypted Document"
                status_badge_color = "#EF4444"
                status_badge_bg = "rgba(239, 68, 68, 0.15)"
            elif extraction_error == "__ERROR_EMPTY_DOCUMENT__" or (file_size_mb == 0 and doc_word_count == 0):
                status_badge_text = "⚠️ Empty Document"
                status_badge_color = "#EF4444"
                status_badge_bg = "rgba(239, 68, 68, 0.15)"
            elif doc_word_count > 40:
                status_badge_text = "✓ Text Extracted & Ready"
                status_badge_color = "#34D399"
                status_badge_bg = "rgba(16, 185, 129, 0.15)"
            elif doc_word_count > 0:
                status_badge_text = "⚠️ Insufficient Academic Text"
                status_badge_color = "#FBBF24"
                status_badge_bg = "rgba(245, 158, 11, 0.15)"
            else:
                status_badge_text = "⚠️ Ready to Extract"
                status_badge_color = "#FBBF24"
                status_badge_bg = "rgba(245, 158, 11, 0.15)"
            
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11.5px; font-weight: 700; color: #38BDF8; letter-spacing: 0.05em; text-transform: uppercase;">📋 Document Extraction Profile</span>
                        <span style="background: {status_badge_bg}; color: {status_badge_color}; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid {status_badge_color}44;">{status_badge_text}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; font-size: 13px;">
                        <div><span style="color: #94A3B8;">Document:</span> <b style="color: #F8FAFC;">{preview_title}</b></div>
                        <div><span style="color: #94A3B8;">Format:</span> <b style="color: #A78BFA;">{ext} ({file_size_mb:.2f} MB)</b></div>
                        <div><span style="color: #94A3B8;">Extracted Words:</span> <b style="color: #34D399;">{f'{doc_word_count:,}' if doc_word_count > 0 else ('0 words (Empty)' if (file_size_mb == 0 or extraction_error == '__ERROR_EMPTY_DOCUMENT__') else 'Ready to parse')}</b></div>
                        <div><span style="color: #94A3B8;">Estimated Pages:</span> <b style="color: #F8FAFC;">{f'~{doc_page_count} pages' if doc_page_count > 0 else '0 pages'}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Step 4: Choose the summary type
            col_mode, col_info = st.columns([1.3, 2.1])
            with col_mode:
                mode_options = ["Short", "Medium", "Long"]
                current_mode_idx = mode_options.index(st.session_state.get("summary_length", "Medium")) if st.session_state.get("summary_length", "Medium") in mode_options else 1
                selected_mode = st.radio(
                    "Choose Summary Detail Level",
                    options=mode_options,
                    index=current_mode_idx,
                    horizontal=True,
                    key="radio_summary_length_choice"
                )
                st.session_state["summary_length"] = selected_mode
            
            with col_info:
                if selected_mode == "Short":
                    st.info("💡 **Short Summary (~100–150 words):** Quick executive overview covering research problem, primary objective, core methodology, key quantitative finding, and main conclusion.")
                elif selected_mode == "Medium":
                    st.info("💡 **Medium Summary (~250–400 words):** Structured thematic synthesis covering background, research objectives, study design/datasets, principal empirical findings, practical implications, and reported limitations.")
                else:
                    st.info("💡 **Long Summary (~600–900 words):** In-depth section-by-section breakdown across all 8 academic pillars: motivation, problem statement, methodology, experimental setup, quantitative results, discussion, validity threats, and future directions.")
            
            btn_sum_click = st.button("📝 Generate Summary", type="primary", use_container_width=True, key="btn_execute_summarization")
            
            if btn_sum_click:
                with st.status("Processing document and generating summary...", expanded=True) as sum_status:
                    try:
                        sum_status.write("✓ Stage 1: Validating document and extracting readable text...")
                        if not st.session_state.get("summary_doc_text"):
                            raw_text = extract_text_from_file(uploaded_file)
                        else:
                            raw_text = st.session_state.summary_doc_text
                            
                        if raw_text == "__ERROR_SCANNED_PDF__":
                            sum_status.update(label="Scanned PDF detected", state="error")
                            st.error("This document appears to be a scanned or image-based PDF with no extractable text layer. Please use an OCR tool (e.g. Adobe Acrobat OCR) to convert the document into searchable text, or upload in DOCX or TXT format.")
                        elif raw_text == "__ERROR_PASSWORD_PROTECTED__":
                            sum_status.update(label="Password-protected document", state="error")
                            st.error("This document is password-protected or encrypted. Please remove password protection before uploading.")
                        elif raw_text == "__ERROR_FILE_TOO_LARGE__":
                            sum_status.update(label="File exceeds size limit", state="error")
                            st.error("The uploaded file exceeds the 30 MB size limit.")
                        elif raw_text == "__ERROR_EMPTY_DOCUMENT__" or not raw_text:
                            sum_status.update(label="Empty document", state="error")
                            st.error("The uploaded document is empty or does not contain readable text.")
                        else:
                            cleaned_doc = clean_text(raw_text)
                            st.session_state.summary_doc_text = cleaned_doc
                            
                            sum_status.write("✓ Stage 2: Segmenting into academic sections & tracking boundaries...")
                            sum_status.write("✓ Stage 3: Scoring sentence salience & empirical evidence...")
                            sum_status.write("✓ Stage 4: Synthesizing cross-section discourse & removing redundancies...")
                            sum_status.write("✓ Stage 5: Generating source-grounded summary...")
                            
                            sum_result = summarizer.summarize_research_paper(
                                text=cleaned_doc,
                                summary_type=selected_mode,
                                filename=uploaded_file.name,
                                doc_title=preview_title
                            )
                            
                            if sum_result.get("success"):
                                st.session_state.summary_result = sum_result
                                st.session_state.summary_doc_name = uploaded_file.name
                                st.session_state.summary_length = selected_mode
                                sum_status.update(label="Summary generated successfully!", state="complete", expanded=False)
                                st.success(f"{selected_mode} summary generated successfully! ({sum_result['word_count']} words)")
                            else:
                                sum_status.update(label="Summarization failed", state="error")
                                st.error(sum_result.get("error", "Summarization failed."))
                    except Exception as sum_err:
                        print(f"[ERROR] Summarization error: {sum_err}")
                        sum_status.update(label="Summarization could not be completed", state="error")
                        st.error("Summarization could not be completed. Please ensure the document contains readable academic text and retry.")

            # Display generated summary if available
            cur_sum = st.session_state.get("summary_result")
            if cur_sum and cur_sum.get("success") and st.session_state.get("summary_doc_name") == uploaded_file.name:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container(border=True):
                    # Summary Header
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px;">
                            <div>
                                <span style="font-size: 11px; font-weight: 700; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">📄 DOCUMENT SUMMARY</span>
                                <h3 style="margin: 2px 0 0 0; font-size: 20px; color: #FFFFFF;">{cur_sum['document_info'].get('title', preview_title)}</h3>
                            </div>
                            <div style="display: flex; gap: 6px; flex-wrap: wrap; align-items: center;">
                                <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 6px; padding: 3px 8px; font-size: 11.5px; color: #C4B5FD; font-weight: 700;">{cur_sum['summary_type']} Summary</span>
                                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 6px; padding: 3px 8px; font-size: 11.5px; color: #34D399; font-weight: 600;">{cur_sum['word_count']} words</span>
                                <span style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 6px; padding: 3px 8px; font-size: 11.5px; color: #38BDF8; font-weight: 600;">~{cur_sum['reading_time_min']} min read</span>
                                <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 3px 8px; font-size: 11.5px; color: #FBBF24; font-weight: 600;">{cur_sum['engine_info'].get('coverage_percentage', 100)}% Coverage</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if cur_sum.get("warnings"):
                        for warn in cur_sum["warnings"]:
                            st.caption(f"ℹ️ Note: {warn}")
                    
                    # Formatted Summary Content
                    st.markdown(cur_sum["summary"])
                    
                    # Grounding & Source Attribution Expander
                    if cur_sum.get("source_attribution"):
                        with st.expander("🔍 Source Evidence Attribution & Grounding", expanded=False):
                            st.markdown("<p style='font-size: 12.5px; color: #94A3B8; margin-bottom: 8px;'>The statements in this summary were grounded directly in the following sections of the uploaded manuscript:</p>", unsafe_allow_html=True)
                            for attr in cur_sum["source_attribution"]:
                                st.markdown(
                                    f"""
                                    <div style="background: rgba(15, 23, 42, 0.45); border-left: 3px solid #38BDF8; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                                        <b style="color: #60A5FA; font-size: 12px;">[{attr.get('pillar', 'Key Evidence')}]</b> &nbsp;
                                        <span style="color: #CBD5E1; font-size: 12px;">Section: <i>{attr.get('section', 'General')}</i> (Page ~{attr.get('page', 1)})</span><br>
                                        <span style="color: #94A3B8; font-size: 11.5px; font-style: italic;">"{attr.get('excerpt', '')}"</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 16px 0 12px 0;'>", unsafe_allow_html=True)
                    
                    # Action Bar (Download, Copy, Regenerate)
                    c_act1, c_act2 = st.columns([1.2, 1.2])
                    with c_act1:
                        txt_export_content = summarizer.export_summary_as_txt(cur_sum)
                        txt_export_name = summarizer.get_summary_export_filename(uploaded_file.name, cur_sum["summary_type"])
                        st.download_button(
                            "⬇ Download Summary (.txt)",
                            data=txt_export_content.encode("utf-8"),
                            file_name=txt_export_name,
                            mime="text/plain",
                            use_container_width=True,
                            key="btn_download_summary_txt"
                        )
                    with c_act2:
                        with st.popover("📋 Copy Full Summary Text", use_container_width=True):
                            st.caption("Click the copy icon in the top right corner of the box below:")
                            st.code(cur_sum["summary"], language=None)

        with tab_eval:
            if st.session_state.get("filename") and uploaded_file.name != st.session_state.get("filename"):
                st.info(f"ℹ️ New document selected ('{uploaded_file.name}'). Click '🚀 Analyze Paper' below to analyze this manuscript and replace previous results for '{st.session_state.get('filename')}'.")
            
            has_matching_analysis = (
                ("results" in st.session_state or "active_analysis" in st.session_state)
                and st.session_state.get("filename") == uploaded_file.name
            )
            
            c_btn1, c_btn2 = st.columns([1.5, 1.8])
            with c_btn1:
                run_analysis_clicked = st.button("🚀 Analyze Paper", type="primary", use_container_width=True, key="btn_run_analysis")
            with c_btn2:
                if has_matching_analysis:
                    st.button("📊 View Analysis Results →", type="primary", use_container_width=True, key="btn_redirect_top_active", on_click=navigate_to, args=("📊 Analysis",))
                else:
                    st.button("📊 View Analysis Results →", disabled=True, use_container_width=True, key="btn_redirect_top_disabled", help="Analyze this paper first to view results.")
        
            if run_analysis_clicked:
                # Invalidate previous document analysis state to prevent stale data leakage
                st.session_state.pop("results", None)
                st.session_state.pop("active_analysis", None)
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("current_analysis_id", None)
                st.session_state.pop("overall_score", None)
                st.session_state["chat_history"] = []
            
                with st.status("Analyzing research paper...", expanded=True) as status:
                    try:
                        st.write("✓ Document extraction")
                        raw_text = extract_text_from_file(uploaded_file)
                    
                        if raw_text == "__ERROR_SCANNED_PDF__":
                            status.update(label="Scanned PDF detected", state="error")
                            st.error("This document appears to be a scanned or image-based PDF with no extractable text layer. Please use an OCR tool (such as Adobe Acrobat OCR or Tesseract) to convert the document into searchable text, or upload the source manuscript in DOCX or TXT format.")
                        elif raw_text == "__ERROR_PASSWORD_PROTECTED__":
                            status.update(label="Password-protected document", state="error")
                            st.error("This document is password-protected or encrypted. Please remove password protection or encryption before uploading.")
                        elif raw_text == "__ERROR_FILE_TOO_LARGE__":
                            status.update(label="File exceeds size limit", state="error")
                            st.error("The uploaded file exceeds the 30 MB size limit. Please upload a smaller or compressed document.")
                        elif raw_text == "__ERROR_CORRUPTED_PDF__":
                            status.update(label="Corrupted PDF", state="error")
                            st.error("The PDF file appears to be corrupted or could not be read. Please check the file and try again.")
                        elif raw_text == "__ERROR_CORRUPTED_DOCX__":
                            status.update(label="Corrupted Word document", state="error")
                            st.error("The Word document (.docx) could not be parsed. Please ensure it is a valid, uncorrupted DOCX file.")
                        elif raw_text == "__ERROR_EMPTY_DOCUMENT__" or not raw_text:
                            status.update(label="Analysis failed: empty document", state="error")
                            st.error("The uploaded document is empty or does not contain readable text. Please upload a valid research paper.")
                        else:
                            st.write("✓ Text preprocessing")
                            cleaned = clean_text(raw_text)
                        
                            if not cleaned or len(cleaned.strip()) < 50:
                                status.update(label="Analysis failed: insufficient text", state="error")
                                st.error("This document does not contain enough extractable academic text for scholarly analysis (minimum 50 characters required). Please ensure the file contains research content.")
                            else:
                                st.write("✓ Structural analysis")
                                sections_data = extract_sections(cleaned)
                            
                                st.write("✓ Language analysis")
                                st.write("✓ Argumentation analysis")
                                st.write("✓ Readability analysis")
                                results = analyze_full_document(cleaned)
                            
                                if results:
                                    st.write("✓ Research insight generation")
                                    keywords, _ = extract_keywords_and_domain(cleaned)
                                    clf = classify_research_domain(cleaned)
                                    publisher = detect_publisher(cleaned)
                                    citations = analyze_citations(cleaned)
                                    semantic = calculate_semantic_strength(cleaned)
                                    novelty = novelty_score(cleaned)
                                    extracted_title = extract_paper_title_from_text(cleaned)
                                
                                    display_title = extracted_title if extracted_title else get_clean_display_title(uploaded_file.name, text=cleaned)
                                    clean_fn = get_clean_report_filename(uploaded_file.name)
                                    overall_score = get_overall_score(results)
                                    grade = get_grade(overall_score)
                                
                                    results["title"] = display_title
                                    results["display_title"] = display_title
                                    results["overall_score"] = overall_score
                                    results["keywords"] = keywords
                                    results["domains"] = clf["all_candidates"]
                                    results["domain"] = clf["primary"]
                                    results["domain_confidence"] = clf["confidence"]
                                    results["domain_secondary"] = clf["secondary"]
                                    results["publisher"] = publisher
                                    results["full_text"] = cleaned
                                    results["citation_analysis"] = citations
                                    results["semantic_strength"] = semantic
                                    results["novelty_score"] = novelty
                                    results["ai_feedback"] = generate_research_feedback(results)
                                    results["potential_research_areas"] = clf["all_candidates"]
                                    results["future_directions"] = extract_future_directions(cleaned)
                                    results["methodology_and_findings"] = extract_methodology_and_findings(cleaned, results["sections"])
                                    results["structured_gaps"] = extract_structured_research_gaps(cleaned)
                                    results["structured_contributions"] = extract_key_contributions_structured(cleaned)
                                    results["recommended_journal"] = recommend_journal(clf["all_candidates"])
                                    results["issues"] = detect_paper_issues(cleaned, results["sections"], results["scores"], results["stats"], citations)
                                    results["vocab_improvements"] = extract_vocabulary_improvements(cleaned)
                                    results["section_sentiments"] = compute_section_sentiments(results["sections"])
                                
                                    st.write("✓ Report generation")
                                    pdf_bytes = generate_pdf_report(results, uploaded_file.name)
                                
                                    # Serialize safely for persistent viewing
                                    serializable_res = dict(results)
                                    serializable_res.pop("blob", None)
                                    res_json = json.dumps(serializable_res)
                                
                                    doc_hash = ""
                                    try:
                                        if file_bytes_raw:
                                            doc_hash = hashlib.sha256(file_bytes_raw).hexdigest()[:16]
                                        elif cleaned:
                                            doc_hash = hashlib.sha256(cleaned.encode("utf-8", errors="ignore")).hexdigest()[:16]
                                    except Exception:
                                        doc_hash = f"hash-{int(time.time())}"
                                    
                                    try:
                                        anl_uuid = f"ANL-{int(time.time())}-{uuid.uuid4().hex[:6].upper()}"
                                    except Exception:
                                        anl_uuid = f"ANL-{int(time.time())}-REC"
                                
                                    rec_id, ts, anl_id = auth.save_analysis_record(
                                        username=username,
                                        filename=uploaded_file.name,
                                        domain=results["domain"],
                                        score=overall_score,
                                        pdf_bytes=pdf_bytes,
                                        results_json=res_json,
                                        analysis_id=anl_uuid,
                                        display_title=display_title,
                                        report_filename=clean_fn,
                                        document_hash=doc_hash,
                                        grade=grade,
                                        dimension_scores=results["scores"]
                                    )
                                
                                    # Authoritative session state synchronization
                                    st.session_state["results"] = results
                                    st.session_state["active_analysis"] = results
                                    st.session_state["analysis_result"] = results
                                    st.session_state["filename"] = uploaded_file.name
                                    st.session_state["paper_title"] = display_title
                                    st.session_state["overall_score"] = overall_score
                                    st.session_state["current_analysis_id"] = anl_id
                                    st.session_state["chat_history"] = []
                                    st.session_state.analysis_history[username] = auth.get_user_history(username)
                                
                                    status.update(label="Paper analyzed successfully!", state="complete", expanded=False)
                                    st.success("Paper analyzed successfully! Click below to view the detailed evaluation.")
                                
                                    c_suc1, c_suc2, _ = st.columns([2.2, 1.8, 2.0])
                                    with c_suc1:
                                        st.button("📊 View Analysis Results →", type="primary", use_container_width=True, key="btn_direct_to_vis", on_click=navigate_to, args=("📊 Analysis",))
                                    with c_suc2:
                                        st.download_button("⬇ Download Report", data=pdf_bytes, file_name=clean_fn, mime="application/pdf", key="btn_direct_down_rep", use_container_width=True)
                                else:
                                    status.update(label="Error analyzing document", state="error")
                                    st.error("Something went wrong while analyzing the document. Please check the file and try again.")
                    except Exception as ex:
                        print(f"[ERROR] Analysis failed: {ex}")
                        status.update(label="Analysis could not be completed", state="error")
                        st.error("Analysis could not be completed. Please retry the analysis. If the problem continues, check the uploaded document format.")

    # Compact Active Analysis Status Card (Non-intrusive)
    elif "results" in st.session_state or "active_analysis" in st.session_state:
        res = st.session_state.get("results") or st.session_state.get("active_analysis")
        overall_score = get_overall_score(res)
        composite = overall_score
        composite_score = overall_score
        grade = get_grade(overall_score)
        filename = st.session_state.get("filename", "Paper")
        clean_disp = st.session_state.get("paper_title") or get_clean_display_title(filename, stored_title=res.get("title") if isinstance(res, dict) else None)
        clean_fn = get_clean_report_filename(filename)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            col_res1, col_res2, col_res3 = st.columns([4.0, 2.0, 2.0])
            with col_res1:
                st.markdown(
                    f"""
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">✓ Active Analysis: 📄 {clean_disp}</div>
                    <div style="color: #94A3B8; font-size: 12px; margin-top: 3px;">
                        Domain: <b style="color: #60A5FA;">{res.get('domain', 'General Academic') if isinstance(res, dict) else 'General Academic'}</b> &nbsp;•&nbsp; 
                        Reading Level: <b style="color: #CBD5E1;">{res.get('stats', {}).get('reading_difficulty', 'Standard Academic') if isinstance(res, dict) else 'Standard Academic'}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_res2:
                st.markdown(
                    f"""
                    <div style="font-size: 15px; font-weight: 800; color: #A78BFA; margin-top: 6px;">
                        Score: {overall_score:.1f} / 100 &nbsp;
                        <span style="background: rgba(139, 92, 246, 0.2); color: #C4B5FD; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 8px;">Grade {grade}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_res3:
                st.button("📊 View Analysis Results →", type="primary", use_container_width=True, key="btn_goto_analysis_act", on_click=navigate_to, args=("📊 Analysis",))
            
            with st.expander("📝 Generate / View Executive Summary for this Paper", expanded=False):
                cur_sum = st.session_state.get("summary_result")
                if cur_sum and cur_sum.get("success") and (cur_sum.get("document_info", {}).get("filename") == filename or cur_sum.get("document_info", {}).get("title") == clean_disp):
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;">
                            <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #C4B5FD; font-weight: 700;">{cur_sum['summary_type']} Summary</span>
                            <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #34D399; font-weight: 600;">{cur_sum['word_count']} words</span>
                            <span style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #38BDF8; font-weight: 600;">~{cur_sum['reading_time_min']} min read</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(cur_sum["summary"])
                    c_dl, c_cp = st.columns([1.5, 1.5])
                    with c_dl:
                        txt_c = summarizer.export_summary_as_txt(cur_sum)
                        txt_n = summarizer.get_summary_export_filename(filename, cur_sum["summary_type"])
                        st.download_button("⬇ Download Summary (TXT)", data=txt_c.encode("utf-8"), file_name=txt_n, mime="text/plain", key="btn_dl_active_sum")
                    with c_cp:
                        with st.popover("📋 Copy Summary Text", use_container_width=True):
                            st.code(cur_sum["summary"], language=None)
                else:
                    st.info("No executive summary has been generated for this active paper yet. Choose a summary mode below to generate one immediately without re-uploading:")
                    col_m1, col_m2 = st.columns([1.3, 2.1])
                    with col_m1:
                        sel_act_mode = st.radio("Summary Mode", ["Short", "Medium", "Long"], index=1, horizontal=True, key="sel_act_paper_mode")
                    with col_m2:
                        if sel_act_mode == "Short":
                            st.caption("Quick executive overview (~100–150 words).")
                        elif sel_act_mode == "Medium":
                            st.caption("Structured thematic explanation (~250–400 words).")
                        else:
                            st.caption("Comprehensive section-by-section breakdown (~600–900 words).")
                    if st.button("📝 Generate Summary for Active Paper", type="primary", use_container_width=True, key="btn_gen_active_sum"):
                        with st.spinner("Generating summary..."):
                            full_txt = res.get("full_text", "")
                            if full_txt:
                                act_sum = summarizer.summarize_research_paper(full_txt, summary_type=sel_act_mode, filename=filename, doc_title=clean_disp)
                                st.session_state.summary_result = act_sum
                                st.session_state.summary_doc_text = full_txt
                                st.session_state.summary_doc_name = filename
                                st.session_state.summary_length = sel_act_mode
                                st.rerun()

# ----------------------------------------------------
# PAGE 3: DETAILED ANALYSIS
# ----------------------------------------------------
elif st.session_state.current_page == "📊 Analysis":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Academic Analysis & Performance</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Clear, explainable evaluation of textual quality, logical structure, and argumentative density.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "results" not in st.session_state:
        st.markdown(
            """
            <div class="iq-card" style="text-align: center; padding: 40px !important;">
                <div style="font-size: 32px; margin-bottom: 8px;">📊</div>
                <h4 style="color: #CBD5E1; margin: 0;">No Paper Analyzed Yet</h4>
                <p style="color: #64748B; margin-top: 4px;">Upload and analyze a research paper in <b>Analyze Paper</b> to inspect detailed analytics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.button("Go to Analyze Paper", type="primary", key="btn_goto_analyze", on_click=navigate_to, args=("📄 Analyze Paper",))
    else:
        res = st.session_state["results"]
        scores = res.get("scores", {})
        stats = res.get("stats", {})
        filename = st.session_state.get("filename", "Paper")
        clean_display_title = st.session_state.get("paper_title") or get_clean_display_title(filename, text=res.get("full_text"), stored_title=res.get("title"))
        clean_fn = get_clean_report_filename(filename)
        comp = get_overall_score(res)
        grade = get_grade(comp)
        
        # A. PAPER OVERVIEW
        areas = res.get("potential_research_areas") or res.get("domains") or [res.get("domain", "General Academic")]
        st.markdown(
            f"""
            <div class="iq-card" style="padding: 18px 22px !important; margin-bottom: 18px !important;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Analyzed Document</div>
                        <div style="font-size: 20px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">📄 {clean_display_title}</div>
                        <div style="margin-top: 8px;">
                            <span style="font-size: 12px; color: #94A3B8; font-weight: 600;">Research Domain:</span>&nbsp;
                            {''.join([f"<span class='domain-badge'>{a}</span>" for a in areas[:3]])}
                            <span style="font-size: 12px; color: #94A3B8; font-weight: 600; margin-left: 12px;">Reading Level:</span>&nbsp;
                            <span style="color: #CBD5E1; font-size: 13px; font-weight: 600;">{res.get('stats', {}).get('reading_difficulty', 'Standard Academic')}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 26px; font-weight: 800; color: #A78BFA;">{comp:.1f} <span style="font-size: 14px; color: #94A3B8;">/ 100</span></div>
                        <div style="background: rgba(139, 92, 246, 0.2); color: #C4B5FD; font-size: 13px; font-weight: 700; padding: 2px 10px; border-radius: 10px; display: inline-block; margin-top: 4px;">Grade {grade}</div>
                    </div>
                </div>
                <div style="border-top: 1px solid #1F2937; margin-top: 14px; padding-top: 10px; font-size: 13px; color: #94A3B8;">
                    Overall evaluation based on language quality, structural coherence, argumentation, academic style, and readability.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Executive Summary Quick View / Generation Expander
        with st.expander("📝 Executive Paper Summary (PaperIQ Summarizer)", expanded=False):
            cur_sum = st.session_state.get("summary_result")
            if cur_sum and cur_sum.get("success") and (cur_sum.get("document_info", {}).get("filename") == filename or cur_sum.get("document_info", {}).get("title") == clean_display_title):
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;">
                        <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #C4B5FD; font-weight: 700;">{cur_sum['summary_type']} Summary</span>
                        <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #34D399; font-weight: 600;">{cur_sum['word_count']} words</span>
                        <span style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #38BDF8; font-weight: 600;">~{cur_sum['reading_time_min']} min read</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(cur_sum["summary"])
                c_dl, c_cp = st.columns([1.5, 1.5])
                with c_dl:
                    txt_c = summarizer.export_summary_as_txt(cur_sum)
                    txt_n = summarizer.get_summary_export_filename(filename, cur_sum["summary_type"])
                    st.download_button("⬇ Download Summary (TXT)", data=txt_c.encode("utf-8"), file_name=txt_n, mime="text/plain", key="btn_dl_anl_summary")
                with c_cp:
                    with st.popover("📋 Copy Summary Text", use_container_width=True):
                        st.code(cur_sum["summary"], language=None)
            else:
                st.info("No executive summary has been generated for this paper yet.")
                col_sm1, col_sm2 = st.columns([1.5, 3.0])
                with col_sm1:
                    sel_anl_len = st.selectbox("Summary Mode", ["Short", "Medium", "Long"], index=1, key="sb_anl_sum_mode")
                with col_sm2:
                    st.write("")
                    st.write("")
                    if st.button("📝 Generate Executive Summary Now", type="primary", key="btn_anl_run_quick_sum"):
                        with st.spinner("Generating summary..."):
                            full_txt = res.get("full_text", "")
                            if full_txt:
                                gen_sum = summarizer.summarize_research_paper(full_txt, summary_type=sel_anl_len, filename=filename, doc_title=clean_display_title)
                                st.session_state.summary_result = gen_sum
                                st.session_state.summary_doc_text = full_txt
                                st.session_state.summary_doc_name = filename
                                st.session_state.summary_length = sel_anl_len
                                st.rerun()
        
        # B. QUALITY DIMENSIONS (5 Main Cards Only)
        st.markdown("<h3 style='font-size: 18px; margin: 0 0 12px 0;'>Quality Dimensions</h3>", unsafe_allow_html=True)
        metrics_keys = ["Language Quality", "Structural Coherence", "Argumentation", "Academic Style", "Readability"]
        card_cols = st.columns(5)
        for c, m in zip(card_cols, metrics_keys):
            val = scores.get(m, scores.get(m.split()[0], 0.0))
            rating, color, interp = interpret_score(val, m)
            with c:
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div class="score-card-title">{m}</div>
                        <div class="score-card-val">{val:.1f} <span style="font-size: 13px; color: #64748B;">/ 100</span></div>
                        <div style="font-size: 13px; font-weight: 700; color: {color}; margin: 2px 0;">{rating}</div>
                        <div class="score-card-interp">"{interp}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # C & D. KEY FINDINGS & AREAS REQUIRING ATTENTION
        col_findings, col_attention = st.columns(2)
        
        l_val = scores.get("Language Quality", scores.get("Language", 0.0))
        c_val = scores.get("Structural Coherence", scores.get("Coherence", 0.0))
        a_val = scores.get("Argumentation", scores.get("Reasoning", 0.0))
        s_val = scores.get("Academic Style", scores.get("Sophistication", 0.0))
        r_val = scores.get("Readability", 0.0)
        
        with col_findings:
            with st.container(border=True):
                st.markdown("<h4 style='color: #34D399; margin: 0 0 12px 0;'>✓ Key Findings</h4>", unsafe_allow_html=True)
                findings_list = []
                if l_val >= 70:
                    findings_list.append("<b>Strong academic vocabulary:</b> Clear sentence structure with formal scholarly cadence.")
                else:
                    findings_list.append("<b>Consistent prose rhythm:</b> Sentences maintain an understandable scholarly cadence.")
                    
                if c_val >= 65:
                    findings_list.append("<b>Consistent structural organization:</b> Logical transitions between sections and topic sentences.")
                else:
                    findings_list.append("<b>Structured thematic blocks:</b> Paper maintains recognizable topical organization.")
                    
                if a_val >= 65:
                    findings_list.append("<b>Evidence-supported arguments:</b> Causal propositions are linked to empirical observations.")
                if s_val >= 65:
                    findings_list.append("<b>Lexical sophistication:</b> Effective use of domain-specific scholarly nomenclature.")
                if r_val >= 60:
                    findings_list.append("<b>Accessible academic readability:</b> Phrasing is well-calibrated for scholarly peer review.")
                    
                for item in findings_list[:4]:
                    st.markdown(f"<p style='font-size: 13px; margin: 0 0 8px 0; color: #CBD5E1;'>✓ {item}</p>", unsafe_allow_html=True)
            
        with col_attention:
            with st.container(border=True):
                st.markdown("<h4 style='color: #F87171; margin: 0 0 12px 0;'>⚠️ Areas Requiring Attention</h4>", unsafe_allow_html=True)
                issues_found = res.get("issues", [])
                if issues_found:
                    for idx, iss in enumerate(issues_found[:3]):
                        category = html.escape(str(iss.get("category", "General")))
                        prob_text = html.escape(str(iss.get("problem", iss.get("issue", "Issue identified"))))
                        why_text = html.escape(str(iss.get("why_it_matters", "Essential for peer-review clarity and scholarly rigor.")))
                        act_text = html.escape(str(iss.get("recommendation", "Review and revise phrasing.")))
                        border_div = "<div style='border-bottom: 1px solid #1F2937; margin: 10px 0;'></div>" if idx < len(issues_found[:3]) - 1 else ""
                        item_html = (
                            f"<div style='margin-bottom: 4px;'>"
                            f"<div style='font-size: 11px; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.05em;'>{category}</div>"
                            f"<div style='font-size: 13px; font-weight: 700; color: #FFFFFF; margin: 2px 0 4px 0;'>{prob_text}</div>"
                            f"<div style='font-size: 12px; color: #94A3B8; margin-bottom: 3px;'><b style='color: #CBD5E1;'>Why it matters:</b> {why_text}</div>"
                            f"<div style='font-size: 12px; color: #CBD5E1;'><b style='color: #A78BFA;'>Action:</b> {act_text}</div>"
                            f"</div>"
                            f"{border_div}"
                        )
                        st.markdown(item_html, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size: 13px; color: #34D399; margin: 0;'>✓ No critical structural, citation, or readability issues detected in the manuscript.</p>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # E. VISUAL ANALYSIS (Radar + Score Breakdown)
        st.markdown("<h3 style='font-size: 18px; margin: 0 0 12px 0;'>Visual Analysis</h3>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        chart_vals = [scores.get(k, scores.get(k.split()[0], 0)) for k in metrics_keys]
        
        with col_c1:
            st.markdown(
                """
                <div style="margin-bottom: 8px;">
                    <h4 style="color: #FFFFFF; margin: 0 0 4px 0;">🎯 Academic Performance Radar</h4>
                    <p style="color: #94A3B8; font-size: 12px; margin: 0;">Relative score distribution across the 5 evaluation dimensions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            ctheme = get_chart_theme(st.session_state.get("theme_preference", "System"))
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=chart_vals + [chart_vals[0]],
                theta=metrics_keys + [metrics_keys[0]],
                fill='toself',
                fillcolor=ctheme["radar_fill"],
                line=dict(color=ctheme["radar_line"], width=2.5),
                hoverinfo="theta+r"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor=ctheme["polar_bg"],
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor=ctheme["grid_color"], tickfont=dict(color=ctheme["tick_color"], size=10)),
                    angularaxis=dict(tickfont=dict(color=ctheme["font_color"], size=11), linecolor=ctheme["line_color"])
                ),
                paper_bgcolor=ctheme["paper_bgcolor"],
                plot_bgcolor=ctheme["plot_bgcolor"],
                font=dict(color=ctheme["font_color"]),
                margin=dict(t=35, b=35, l=45, r=45),
                height=360,
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_c2:
            st.markdown(
                """
                <div style="margin-bottom: 8px;">
                    <h4 style="color: #FFFFFF; margin: 0 0 4px 0;">📊 Score Breakdown</h4>
                    <p style="color: #94A3B8; font-size: 12px; margin: 0;">Component dimension scores compared against composite score.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            bar_labels = ["Language<br>Quality", "Structural<br>Coherence", "Argumentation", "Academic<br>Style", "Readability", "Overall<br>Score"]
            bar_vals = chart_vals + [comp]
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=bar_labels,
                    y=bar_vals,
                    marker=dict(color=["#8B5CF6", "#6366F1", "#3B82F6", "#06B6D4", "#10B981", "#A855F7"]),
                    text=[f"{v:.1f}" for v in bar_vals],
                    textposition="auto",
                    textfont=dict(color="#FFFFFF", size=11)
                )
            ])
            fig_bar.update_layout(
                paper_bgcolor=ctheme["paper_bgcolor"],
                plot_bgcolor=ctheme["plot_bgcolor"],
                font=dict(color=ctheme["font_color"]),
                yaxis=dict(range=[0, 105], gridcolor=ctheme["grid_color"], linecolor=ctheme["line_color"], title="Score (0-100)"),
                xaxis=dict(linecolor=ctheme["line_color"], tickfont=dict(color=ctheme["tick_color"], size=11)),
                margin=dict(t=35, b=35, l=35, r=35),
                height=360
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # F. DOCUMENT STATISTICS (Collapsible / Expandable Section)
        with st.expander("📊 Document Statistics", expanded=False):
            dm1, dm2, dm3, dm4, dm5, dm6 = st.columns(6)
            dm1.metric("Total Words", f"{stats.get('word_count', 0):,}")
            dm2.metric("Total Sentences", f"{stats.get('sentence_count', 0):,}")
            dm3.metric("Avg Sentence Len", f"{stats.get('avg_sentence_len', 0):.1f}")
            dm4.metric("Avg Word Len", f"{stats.get('avg_word_len', 0):.1f}")
            dm5.metric("Vocab Diversity", f"{stats.get('vocab_diversity', 0):.2f}")
            dm6.metric("Complex Word Ratio", f"{stats.get('complex_word_ratio', 0):.2f}")
            
        # G. CLEAN SECTION SUMMARIES (Only if Reliable Academic Sections Exist)
        reliable_secs = get_reliable_academic_sections(res.get("sections", {}))
        if reliable_secs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Section Summaries")
            for sec_name, sec_content in reliable_secs.items():
                with st.expander(f"📑 {sec_name}", expanded=True if sec_name in ["Abstract", "Introduction"] else False):
                    summary_text = summarize_text(sec_content, 3)
                    key_points = get_important_sentences(summary_text, 2)
                    st.markdown(f"**Executive Summary:**<br>{summary_text}", unsafe_allow_html=True)
                    if key_points:
                        st.markdown("<br>**Key Observations:**", unsafe_allow_html=True)
                        for kp in key_points:
                            st.markdown(f"• {kp}")

# ----------------------------------------------------
# PAGE 4: INSIGHTS & IMPROVEMENTS
# ----------------------------------------------------
elif st.session_state.current_page == "💡 Insights":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Research Insights & Writing Improvements</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Actionable evaluation of research opportunities, contributions, and structural writing refinements.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "results" not in st.session_state:
        st.markdown(
            """
            <div class="iq-card" style="text-align: center; padding: 40px !important;">
                <div style="font-size: 32px; margin-bottom: 8px;">💡</div>
                <h4 style="color: #CBD5E1; margin: 0;">No Insights Generated Yet</h4>
                <p style="color: #64748B; margin-top: 4px;">Analyze a research paper to generate scholarly insights and recommendations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.button("Go to Analyze Paper", type="primary", key="btn_insights_goto_analyze", on_click=navigate_to, args=("📄 Analyze Paper",))
    else:
        res = st.session_state["results"]
        full_text = res.get("full_text", "")
        scores = res.get("scores", {})
        stats = res.get("stats", {})
        
        # 1. Potential Research Areas & Domain Context
        areas_list = res.get("potential_research_areas") or res.get("domains") or [res.get("domain", "General Academic")]
        st.markdown(
            f"""
            <div class="journal-hero-card">
                <h4>📚 Potential Research Areas</h4>
                <div style="margin: 8px 0 10px 0;">
                    {''.join([f"<span class='domain-badge'>{d}</span>" for d in areas_list])}
                </div>
                <div style="font-size: 12px; color: #94A3B8; border-top: 1px solid rgba(139, 92, 246, 0.25); padding-top: 8px;">
                    🛡️ <b>Disclaimer:</b> Inferred based on document terminology and computational keyword modeling. Not an official journal endorsement or indexing guarantee.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # RESEARCH INSIGHTS (Gaps & Contributions)
        st.subheader("Research Insights")
        col_gaps, col_contrib = st.columns(2)
        
        with col_gaps:
            with st.container(border=True):
                st.markdown("<h4 style='color: #60A5FA; margin: 0 0 12px 0;'>🔍 Research Gaps & Opportunities</h4>", unsafe_allow_html=True)
                structured_gaps = res.get("structured_gaps") or extract_structured_research_gaps(full_text)
                if structured_gaps:
                    for g in structured_gaps:
                        conf_color = "#34D399" if g.get("confidence") == "High" else "#FBBF24"
                        conf_label = html.escape(str(g.get("confidence", "Moderate")))
                        lim_label = html.escape(str(g.get("limitation", "Identified Constraint")))
                        ev_text = html.escape(str(g.get("evidence", "")))
                        opp_text = html.escape(str(g.get("opportunity", "")))
                        gap_card_html = (
                            f"<div style='background: rgba(30, 41, 59, 0.7); border: 1px solid #374151; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; word-break: break-word;'>"
                            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>"
                            f"<span style='font-size: 11px; font-weight: 700; color: #60A5FA; text-transform: uppercase;'>{lim_label}</span>"
                            f"<span style='font-size: 10px; font-weight: 700; color: {conf_color}; background: rgba(255,255,255,0.06); padding: 2px 7px; border-radius: 4px; border: 1px solid {conf_color}40;'>Confidence: {conf_label}</span>"
                            f"</div>"
                            f"<div style='color: #CBD5E1; font-style: italic; font-size: 12px; margin: 6px 0; line-height: 1.5;'>&ldquo;{ev_text}&rdquo;</div>"
                            f"<div style='color: #94A3B8; font-size: 12px; margin-top: 6px;'><b style='color: #A78BFA;'>Research Opportunity:</b> {opp_text}</div>"
                            f"</div>"
                        )
                        st.markdown(gap_card_html, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #94A3B8; font-size: 13px; font-style: italic; padding: 8px 0;'>A specific research gap could not be established from the available document content. The manuscript does not contain explicit limitation statements or unresolved inquiry markers.</div>", unsafe_allow_html=True)
            
        with col_contrib:
            with st.container(border=True):
                st.markdown("<h4 style='color: #34D399; margin: 0 0 12px 0;'>🎯 Key Contributions</h4>", unsafe_allow_html=True)
                structured_contribs = res.get("structured_contributions") or extract_key_contributions_structured(full_text)
                if structured_contribs:
                    for c in structured_contribs:
                        is_author = c.get("type") == "Author-Stated Contribution"
                        badge_color = "#34D399" if is_author else "#60A5FA"
                        type_label = html.escape(str(c.get("type", "Key Contribution")))
                        conf_label = html.escape(str(c.get("confidence", "Moderate")))
                        c_text = html.escape(str(c.get("text", "")))
                        contrib_card_html = (
                            f"<div style='background: rgba(30, 41, 59, 0.7); border: 1px solid #374151; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; word-break: break-word;'>"
                            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>"
                            f"<span style='font-size: 11px; font-weight: 700; color: {badge_color}; text-transform: uppercase;'>{type_label}</span>"
                            f"<span style='font-size: 10px; font-weight: 700; color: #CBD5E1; background: rgba(255,255,255,0.06); padding: 2px 7px; border-radius: 4px;'>Confidence: {conf_label}</span>"
                            f"</div>"
                            f"<div style='color: #F8FAFC; font-size: 13px; margin-top: 4px; line-height: 1.5;'>&ldquo;{c_text}&rdquo;</div>"
                            f"</div>"
                        )
                        st.markdown(contrib_card_html, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #94A3B8; font-size: 13px; font-style: italic; padding: 8px 0;'>No explicit author contribution statements could be established from the analyzed manuscript text.</div>", unsafe_allow_html=True)
            
        # METHODOLOGY, DATA & EMPIRICAL FINDINGS
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Methodology, Data & Empirical Findings")
        meth_findings = res.get("methodology_and_findings") or extract_methodology_and_findings(full_text, res.get("sections", {}))
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            with st.container(border=True):
                st.markdown("<h4 style='color: #60A5FA; margin: 0 0 10px 0;'>🔬 Methodology & Study Design</h4>", unsafe_allow_html=True)
                m_txt = html.escape(str(meth_findings.get('methodology', 'Not explicitly reported in the analyzed text.')))
                st.markdown(f"<p style='font-size: 13px; color: #CBD5E1; line-height: 1.6; margin: 0;'>{m_txt}</p>", unsafe_allow_html=True)
        with c_m2:
            with st.container(border=True):
                st.markdown("<h4 style='color: #F59E0B; margin: 0 0 10px 0;'>📦 Data Sources & Materials</h4>", unsafe_allow_html=True)
                d_txt = html.escape(str(meth_findings.get('data_and_materials', 'Not explicitly reported in the analyzed text.')))
                st.markdown(f"<p style='font-size: 13px; color: #CBD5E1; line-height: 1.6; margin: 0;'>{d_txt}</p>", unsafe_allow_html=True)
        with c_m3:
            with st.container(border=True):
                st.markdown("<h4 style='color: #34D399; margin: 0 0 10px 0;'>📊 Empirical Findings & Outcomes</h4>", unsafe_allow_html=True)
                f_txt = html.escape(str(meth_findings.get('findings', 'Not explicitly reported in the analyzed text.')))
                st.markdown(f"<p style='font-size: 13px; color: #CBD5E1; line-height: 1.6; margin: 0;'>{f_txt}</p>", unsafe_allow_html=True)
            
        # FUTURE RESEARCH DIRECTIONS
        future_dirs = res.get("future_directions") or extract_future_directions(full_text)
        has_author_stated = any("not enough evidence" not in fd.lower() for fd in future_dirs)
        
        with st.container(border=True):
            st.markdown("<h4 style='color: #A78BFA; margin: 0 0 12px 0;'>🔭 Future Research Directions</h4>", unsafe_allow_html=True)
            if has_author_stated:
                for fd in future_dirs:
                    if "not enough evidence" not in fd.lower():
                        fd_clean = html.escape(str(fd))
                        fd_card_html = (
                            f"<div style='background: rgba(30, 41, 59, 0.7); border: 1px solid #374151; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; word-break: break-word;'>"
                            f"<span style='font-size: 11px; font-weight: 700; color: #A78BFA; text-transform: uppercase;'>Author-Stated Future Work:</span>"
                            f"<div style='color: #F8FAFC; font-size: 13px; margin-top: 4px; line-height: 1.5;'>&ldquo;{fd_clean}&rdquo;</div>"
                            f"</div>"
                        )
                        st.markdown(fd_card_html, unsafe_allow_html=True)
            else:
                fd_card_html = (
                    f"<div style='background: rgba(30, 41, 59, 0.7); border: 1px solid #374151; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;'>"
                    f"<span style='font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Analytical Suggestion (Derived from evaluation, not claimed by authors):</span>"
                    f"<div style='color: #CBD5E1; font-size: 13px; margin-top: 4px; line-height: 1.5;'>Conduct external validation experiments and benchmark against standardized baselines to establish broader domain generalizability.</div>"
                    f"</div>"
                )
                st.markdown(fd_card_html, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # WRITING IMPROVEMENTS (Readability, Structural, Argumentation)
        st.subheader("Writing Improvements")
        
        # Readability Improvements
        long_sentences = extract_overly_long_sentences(full_text, max_words=35, limit=2)
        with st.container(border=True):
            st.markdown("<h4 style='color: #FFFFFF; margin: 0 0 12px 0;'>📖 Readability Improvements</h4>", unsafe_allow_html=True)
            if long_sentences:
                for s_text, word_cnt in long_sentences:
                    clean_sample = html.escape(str(s_text))
                    r_card_html = (
                        f"<div style='background: rgba(30, 41, 59, 0.7); border: 1px solid #374151; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; word-break: break-word;'>"
                        f"<div style='color: #FBBF24; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Complex Sentence Identified ({word_cnt} words):</div>"
                        f"<div style='color: #CBD5E1; font-style: italic; font-size: 13px; margin: 6px 0; line-height: 1.5;'>&ldquo;{clean_sample}&rdquo;</div>"
                        f"<div style='color: #94A3B8; font-size: 12px; margin-top: 6px;'><b style='color: #A78BFA;'>Recommendation:</b> Break this compound sentence into two concise statements to improve readability and reviewer comprehension.</div>"
                        f"</div>"
                    )
                    st.markdown(r_card_html, unsafe_allow_html=True)
            else:
                avg_len = stats.get('avg_sentence_len', 22) if isinstance(stats, dict) else 22
                st.markdown(f"<p style='color: #34D399; font-size: 13px; margin: 0;'>✓ Sentence lengths are well-calibrated (average: {avg_len:.1f} words). Phrasing remains accessible for scholarly evaluation.</p>", unsafe_allow_html=True)
        
        # Structural & Argumentation Improvements
        col_struct, col_arg = st.columns(2)
        with col_struct:
            c_score = scores.get("Structural Coherence", scores.get("Coherence", 0.0))
            with st.container(border=True):
                st.markdown("<h4 style='color: #FFFFFF; margin: 0 0 6px 0;'>🏗️ Structural Improvements</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 13px; color: #CBD5E1; margin-bottom: 8px;'>Coherence Index: <b>{c_score:.1f} / 100</b></p>", unsafe_allow_html=True)
                rec_struct = 'Incorporate directional discourse markers (e.g., "Consequently", "In contrast", "Furthermore") between subsection transitions to improve thematic flow.' if c_score < 70 else 'Discourse connectors are consistently deployed across major argument blocks; ensure paragraph openers state clear thematic topic sentences.'
                st.markdown(f"<p style='font-size: 13px; color: #94A3B8; margin: 0;'>{rec_struct}</p>", unsafe_allow_html=True)
            
        with col_arg:
            a_score = scores.get("Argumentation", scores.get("Reasoning", 0.0))
            with st.container(border=True):
                st.markdown("<h4 style='color: #FFFFFF; margin: 0 0 6px 0;'>⚖️ Argumentation Improvements</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 13px; color: #CBD5E1; margin-bottom: 8px;'>Argumentation Index: <b>{a_score:.1f} / 100</b></p>", unsafe_allow_html=True)
                rec_arg = 'Directly couple causal propositions with explicit empirical evidence indicators (e.g., "demonstrates that", "substantiates", "implies").' if a_score < 70 else 'Causal assertions are well-grounded by supporting evidence and empirical outcomes throughout findings.'
                st.markdown(f"<p style='font-size: 13px; color: #94A3B8; margin: 0;'>{rec_arg}</p>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # RECOMMENDED ACTIONS (Clear, Practical Next Steps)
        st.subheader("Recommended Next Actions")
        actions = []
        if any(iss.get("category") == "Structure" for iss in res.get("issues", [])):
            actions.append("<b>Clarify Methodology:</b> Add or expand a dedicated Methodology section outlining research design, dataset specifications, and evaluation procedure.")
        if any(iss.get("category") == "Citations" for iss in res.get("issues", [])):
            actions.append("<b>Improve Citation Coverage:</b> Anchor empirical findings and related work within authoritative peer-reviewed literature.")
        if long_sentences:
            actions.append("<b>Split Overly Long Sentences:</b> Segment compound sentences exceeding 35 words into concise propositions to improve readability.")
        if c_score < 70:
            actions.append("<b>Improve Structural Transitions:</b> Use directional discourse markers between consecutive sections to guide reader flow.")
        if a_score < 70:
            actions.append("<b>Strengthen Empirical Evidence:</b> Ensure all primary claims link directly to experimental data, benchmark metrics, or baseline comparisons.")
        if not actions:
            actions.append("<b>Camera-Ready Polish:</b> Polish abstract and conclusion sections to highlight principal empirical findings and future research directions.")
            actions.append("<b>Consistency Check:</b> Verify consistent notation and formatting across all equations, figures, and table captions.")
            
        with st.container(border=True):
            items_html = "".join([f"<li style='margin-bottom: 6px;'>{act}</li>" for act in actions[:4]])
            st.markdown(f"<ol style='color: #CBD5E1; font-size: 14px; padding-left: 20px; margin: 0; line-height: 1.8;'>{items_html}</ol>", unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE 5: ASK PAPERIQ (DOCUMENT Q&A)
# ----------------------------------------------------
elif st.session_state.current_page == "💬 Ask PaperIQ":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Ask PaperIQ</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Ask questions about your research paper. PaperIQ answers questions grounded exclusively in the currently analyzed document.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "results" not in st.session_state:
        st.markdown(
            """
            <div class="iq-card" style="text-align: center; padding: 40px !important;">
                <div style="font-size: 32px; margin-bottom: 8px;">💬</div>
                <h4 style="color: #CBD5E1; margin: 0;">Analyze a Paper to Start Q&A</h4>
                <p style="color: #64748B; margin-top: 4px;">Upload and analyze a research paper first to chat with its contents.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.button("Go to Analyze Paper", type="primary", key="btn_ask_goto_analyze", on_click=navigate_to, args=("📄 Analyze Paper",))
    else:
        res = st.session_state["results"]
        full_text = res.get("full_text", "")
        sections = res.get("sections", {})
        active_doc = st.session_state.get("paper_title") or get_clean_display_title(st.session_state.get("filename", "Current Document"), stored_title=res.get("title"))
        domain_name = res.get("domain", "General Academic")
        overall_score = get_overall_score(res)
        
        col_ctx1, col_ctx2 = st.columns([5, 1.2])
        with col_ctx1:
            st.markdown(
                f"""
                <div class="iq-card" style="padding: 14px 18px !important; margin-bottom: 16px !important;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.05em;">Current Paper</div>
                    <div style="font-size: 17px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">📄 {active_doc}</div>
                    <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">
                        Domain: <b style="color: #60A5FA;">{domain_name}</b> &nbsp;•&nbsp; 
                        Score: <b style="color: #A78BFA;">{overall_score:.1f} / 100</b> &nbsp;•&nbsp; 
                        <span style="color: #34D399; font-weight: 600;">● Document loaded</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_ctx2:
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat", key="btn_clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # 11 Standardized Suggested Questions
        st.markdown("<p style='color: #94A3B8; font-size: 13px; margin-bottom: 8px; font-weight: 600;'>Suggested Questions:</p>", unsafe_allow_html=True)
        suggested = [
            "What is the main objective of this paper?",
            "What methodology was used?",
            "What are the key findings?",
            "What are the limitations?",
            "What research gap does this paper address?",
            "What datasets or materials were used?",
            "How is this work evaluated?",
            "What are the core contributions?",
            "What future work is suggested?",
            "What theoretical framework is applied?",
            "Summarize the conclusion."
        ]
        
        sug_cols = st.columns(3)
        for i, q_text in enumerate(suggested):
            target_col = sug_cols[i % 3]
            with target_col:
                if st.button(q_text, key=f"sug_btn_{i}", use_container_width=True):
                    ans_text, sources = grounded_paper_qa(q_text, full_text, sections)
                    clean_src = ""
                    if sources and not ans_text.startswith("I couldn't find enough evidence"):
                        clean_sec_name = sources[0].replace("Section: ", "").strip()
                        clean_src = f"{clean_sec_name} section"
                    st.session_state.chat_history.append({"role": "user", "content": q_text})
                    st.session_state.chat_history.append({"role": "assistant", "answer": ans_text, "source": clean_src})
                    st.rerun()
                    
        st.markdown("<hr style='border-color: #1F2937; margin: 16px 0 14px 0;'>", unsafe_allow_html=True)
        
        # Render Chat History
        if not st.session_state.chat_history:
            st.markdown("<div style='color: #64748B; font-size: 13px; font-style: italic; margin-bottom: 12px;'>No questions asked yet. Click a suggested question above or enter a question below.</div>", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg.get("role") == "user":
                    user_c = html.escape(str(msg.get('content', '')))
                    st.markdown(f"<div class='chat-msg-user'><b>👤 You:</b><div style='margin-top: 4px; color: #FFFFFF;'>{user_c}</div></div>", unsafe_allow_html=True)
                else:
                    ans_body = html.escape(str(msg.get("answer") or msg.get("content", "")))
                    src_val = html.escape(str(msg.get("source", "")))
                    source_pill_html = f"<div style='background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 12px; color: #CBD5E1; display: inline-block; margin-top: 6px;'><b>Evidence:</b> {src_val}</div>" if src_val else ""
                    bot_html = (
                        f"<div class='chat-msg-bot'>"
                        f"<div style='font-weight: 700; color: #A78BFA; font-size: 14px; margin-bottom: 6px;'>🤖 PaperIQ</div>"
                        f"<div style='color: #F8FAFC; font-size: 14px; line-height: 1.6; margin-bottom: 6px; word-break: break-word;'>{ans_body}</div>"
                        f"{source_pill_html}"
                        f"</div>"
                    )
                    st.markdown(bot_html, unsafe_allow_html=True)
                    
        # Compact Inline Chat Input (Eliminates huge blank bottom gap!)
        with st.form(key="ask_form", clear_on_submit=True):
            col_inp, col_snd = st.columns([5, 1])
            with col_inp:
                user_query = st.text_input("Ask about this paper...", placeholder="Ask a question about the analyzed paper...", label_visibility="collapsed")
            with col_snd:
                send_pressed = st.form_submit_button("Send", type="primary", use_container_width=True)
                
            if send_pressed and user_query and user_query.strip():
                ans_text, sources = grounded_paper_qa(user_query.strip(), full_text, sections)
                clean_src = ""
                if sources and not ans_text.startswith("I couldn't find enough evidence"):
                    clean_sec_name = sources[0].replace("Section: ", "").strip()
                    clean_src = f"{clean_sec_name} section"
                st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
                st.session_state.chat_history.append({"role": "assistant", "answer": ans_text, "source": clean_src})
                st.rerun()

# ----------------------------------------------------
# PAGE 6: PAPER COMPARISON
# ----------------------------------------------------
elif st.session_state.current_page == "🔄 Compare Papers":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Compare Research Papers</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Compare research papers across writing quality, structure, argumentation, and readability.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="iq-card">
            <h4 style="margin: 0 0 8px 0;">Upload Papers for Comparison</h4>
            <p style="color: #94A3B8; font-size: 13px; margin: 0 0 14px 0;">Upload 2 or 3 research papers (PDF, DOCX, TXT, max 30 MB each) to evaluate writing clarity and structural consistency.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    c_up1, c_up2, c_up3 = st.columns(3)
    with c_up1:
        f1 = st.file_uploader("Paper 1", type=["pdf", "docx", "txt"], key="comp_file_1")
    with c_up2:
        f2 = st.file_uploader("Paper 2", type=["pdf", "docx", "txt"], key="comp_file_2")
    with c_up3:
        f3 = st.file_uploader("Paper 3 (Optional)", type=["pdf", "docx", "txt"], key="comp_file_3")
        
    if st.button("🚀 Compare Papers", type="primary", use_container_width=True, key="btn_run_compare"):
        uploaded_comp = [f for f in [f1, f2, f3] if f is not None]
        if len(uploaded_comp) < 2:
            st.warning("Please upload at least 2 papers to perform comparison.")
        else:
            with st.spinner("Analyzing and comparing papers..."):
                compared_list = []
                failed_files = []
                for f in uploaded_comp:
                    text_c = extract_text_from_file(f)
                    if isinstance(text_c, str) and text_c.startswith("__ERROR_"):
                        failed_files.append((f.name, text_c))
                        continue
                    cleaned_c = clean_text(text_c)
                    if not cleaned_c or len(cleaned_c.strip()) < 50:
                        failed_files.append((f.name, "__ERROR_EMPTY_DOCUMENT__"))
                        continue
                    res_c = analyze_full_document(cleaned_c)
                    if res_c:
                        clf_c = classify_research_domain(cleaned_c)
                        cits = analyze_citations(cleaned_c)
                        res_c["filename"] = f.name
                        res_c["clean_title"] = get_clean_display_title(f.name, text=cleaned_c)
                        res_c["domain"] = clf_c["primary"]
                        res_c["domain_confidence"] = clf_c["confidence"]
                        res_c["citation_analysis"] = cits
                        res_c["full_text"] = cleaned_c
                        compared_list.append(res_c)
                    else:
                        failed_files.append((f.name, "Analysis failed"))
                
                if failed_files:
                    for fname, reason in failed_files:
                        err_msg = "insufficient extractable text"
                        if "SCANNED" in reason:
                            err_msg = "scanned/image-only PDF (OCR required)"
                        elif "PASSWORD" in reason:
                            err_msg = "password-protected document"
                        elif "LARGE" in reason:
                            err_msg = "file too large (>30 MB)"
                        elif "CORRUPTED" in reason:
                            err_msg = "corrupted or unreadable file"
                        st.warning(f"⚠️ Could not process '{fname}': {err_msg}")
                        
                if len(compared_list) < 2:
                    st.error("Comparison requires at least 2 successfully extracted research papers. Please provide valid documents.")
                else:
                    st.session_state["comparison_data"] = compared_list
                    st.success(f"Successfully compared {len(compared_list)} papers!")

    comp_data = st.session_state.get("comparison_data", [])
    if comp_data and len(comp_data) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
                <span style="color: #FFFFFF; font-size: 13px; font-weight: 600;">Comparative Summary</span>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 2px;">Side-by-side comparison across key academic dimensions. Evaluates writing clarity and structure without claiming scientific superiority.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 1. Clean Comparison Table
        dimensions_list = [
            ("Language Quality", "Language Quality"),
            ("Structural Coherence", "Structural Coherence"),
            ("Argumentation", "Argumentation"),
            ("Academic Style", "Academic Style"),
            ("Readability", "Readability"),
            ("Overall", "Composite"),
        ]
        
        comp_table_data = {"Dimension": [d[0] for d in dimensions_list]}
        for p in comp_data:
            p_name = p.get("clean_title", p["filename"])
            s = p["scores"]
            col_vals = []
            for label, key in dimensions_list:
                val = s.get(key, s.get(key.split()[0], 0.0))
                col_vals.append(f"{val:.1f}")
            comp_table_data[p_name] = col_vals
            
        df_comp = pd.DataFrame(comp_table_data)
        st.dataframe(df_comp.set_index("Dimension"), use_container_width=True)
        
        # 2. Best Performer Card
        best_p = max(comp_data, key=lambda p: get_overall_score(p))
        best_title = best_p.get("clean_title", best_p["filename"])
        best_score = get_overall_score(best_p)
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 12px 18px; margin: 14px 0;">
                <div style="color: #34D399; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">★ Highest Composite Writing Score</div>
                <div style="color: #FFFFFF; font-size: 16px; font-weight: 700; margin-top: 2px;">{best_title}</div>
                <div style="color: #A7F3D0; font-size: 13px; margin-top: 2px;">Overall Score: <b>{best_score:.1f} / 100</b> (Grade {get_grade(best_score)})</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 3. Comparative Summary Cards
        st.markdown("<h3 style='font-size: 18px; margin: 16px 0 12px 0;'>Comparative Strengths</h3>", unsafe_allow_html=True)
        
        comp_cols = st.columns(len(comp_data))
        for idx_p, p in enumerate(comp_data):
            with comp_cols[idx_p]:
                p_title = p.get("clean_title", p["filename"])
                s = p["scores"]
                p_ovr = get_overall_score(p)
                l_v = s.get("Language Quality", s.get("Language", 0.0))
                c_v = s.get("Structural Coherence", s.get("Coherence", 0.0))
                a_v = s.get("Argumentation", s.get("Reasoning", 0.0))
                st.markdown(
                    f"""
                    <div class="iq-card">
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 14px; margin-bottom: 4px;">📄 {p_title}</div>
                        <div style="font-size: 12px; color: #A78BFA; font-weight: 700; margin-bottom: 8px;">Overall: {p_ovr:.1f} (Grade {get_grade(p_ovr)})</div>
                        <div style="font-size: 12px; color: #34D399; font-weight: 600;">Strengths:</div>
                        <ul style="font-size: 12px; color: #CBD5E1; padding-left: 16px; margin: 4px 0 8px 0;">
                            <li>{('Strong' if l_v >= 70 else 'Adequate')} syntactic phrasing ({l_v:.1f}/100)</li>
                            <li>{('Consistent' if c_v >= 65 else 'Moderate')} discourse flow ({c_v:.1f}/100)</li>
                            <li>{('Rigorously grounded' if a_v >= 65 else 'Moderate')} argumentation ({a_v:.1f}/100)</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # 4. Comparative Bar Chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig_comp = go.Figure()
        dims = ["Language Quality", "Structural Coherence", "Argumentation", "Academic Style", "Readability", "Overall"]
        for p in comp_data:
            s = p["scores"]
            y_vals = [
                s.get("Language Quality", s.get("Language", 0)),
                s.get("Structural Coherence", s.get("Coherence", 0)),
                s.get("Argumentation", s.get("Reasoning", 0)),
                s.get("Academic Style", s.get("Sophistication", 0)),
                s.get("Readability", 0),
                get_overall_score(p)
            ]
            fig_comp.add_trace(go.Bar(
                name=p.get("clean_title", p["filename"]),
                x=dims,
                y=y_vals
            ))
        ctheme_comp = get_chart_theme(st.session_state.get("theme_preference", "System"))
        fig_comp.update_layout(
            barmode='group',
            paper_bgcolor=ctheme_comp["paper_bgcolor"],
            plot_bgcolor=ctheme_comp["plot_bgcolor"],
            font=dict(color=ctheme_comp["font_color"]),
            yaxis=dict(range=[0, 105], gridcolor=ctheme_comp["grid_color"], linecolor=ctheme_comp["line_color"], title="Score (0-100)"),
            xaxis=dict(linecolor=ctheme_comp["line_color"], tickfont=dict(color=ctheme_comp["tick_color"])),
            legend=dict(font=dict(color=ctheme_comp["font_color"]), bgcolor=ctheme_comp["legend_bg"], bordercolor=ctheme_comp["legend_border"])
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # 5. Recommendation & Limitation Notice
        st.markdown(
            """
            <div class="iq-card" style="margin-top: 14px; border-left: 4px solid #60A5FA !important;">
                <div style="font-size: 13px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">ℹ️ Comparative Evaluation Scope</div>
                <p style="font-size: 12px; color: #94A3B8; margin: 0; line-height: 1.5;">
                    Comparative ratings evaluate textual clarity, rhetorical organization, vocabulary sophistication, and readability. A higher writing score does not guarantee superior scientific correctness, experimental validity, or peer-review acceptance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------------------------------
# PAGE 7: ANALYSIS HISTORY
# ----------------------------------------------------
elif st.session_state.current_page in ["📚 History", "🕘 History"]:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Analysis History</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Persistently saved research evaluations and downloadable analysis reports from your local workspace.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if not user_history:
        st.markdown(
            """
            <div class="iq-card" style="text-align: center; padding: 40px !important;">
                <div style="font-size: 32px; margin-bottom: 8px;">📚</div>
                <h4 style="color: #CBD5E1; margin: 0;">No Analysis History Yet</h4>
                <p style="color: #64748B; margin-top: 4px;">Analyze your first paper in <b>Analyze Paper</b> to archive reports here.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Search and Domain Filtering Controls
        f_col1, f_col2 = st.columns([3, 2])
        with f_col1:
            search_query = st.text_input("🔍 Search papers by title", "", key="hist_search_box")
        with f_col2:
            all_domains = sorted(list(set([item.get("domain", "General Academic") for item in user_history])))
            domain_filter = st.selectbox("Filter by Domain", ["All Domains"] + all_domains, key="hist_domain_filter")
            
        filtered_history = [
            item for item in user_history
            if (not search_query or search_query.lower() in f"{item.get('display_title', '')} {item.get('original_filename', '')} {item.get('filename', '')} {item.get('domain', '')}".lower())
            and (domain_filter == "All Domains" or item.get("domain") == domain_filter)
        ]
        
        st.markdown(f"<p style='color: #94A3B8; font-size: 13px; margin: 6px 0 12px 0;'>Showing <b>{len(filtered_history)}</b> of <b>{len(user_history)}</b> analyses</p>", unsafe_allow_html=True)
        
        if not filtered_history:
            st.info("No analyses match the current search or filter criteria.")
        else:
            for idx, item in enumerate(filtered_history):
                item_score = get_overall_score(item)
                grade = get_grade(item_score)
                res_meta = None
                if item.get("results_json"):
                    try:
                        res_meta = json.loads(item["results_json"])
                    except Exception:
                        pass
                stored_title = item.get("display_title") or (res_meta.get("title") if res_meta else None)
                clean_title = get_clean_display_title(item.get("filename", "Paper"), stored_title=stored_title)
                truncated_title = clean_title if len(clean_title) <= 45 else f"{clean_title[:42]}..."
                clean_fn = get_clean_report_filename(item.get("filename", "Paper"))
                
                with st.container(border=True):
                    col_info, col_score, col_v, col_d, col_del = st.columns([3.8, 1.8, 1.6, 1.6, 1.2])
                    with col_info:
                        st.markdown(
                            f"""
                            <div style="font-size: 15px; font-weight: 700; color: #FFFFFF;" title="{clean_title}">📄 {truncated_title}</div>
                            <div style="font-size: 12px; color: #94A3B8; margin-top: 3px;">
                                Domain: <b style="color: #60A5FA;">{item.get('domain', 'General Academic')}</b> &nbsp;•&nbsp; 
                                Date: {item.get('created_at')}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_score:
                        st.markdown(
                            f"""
                            <div style="font-size: 15px; font-weight: 800; color: #A78BFA; margin-top: 6px;">
                                Score: {item_score:.1f} / 100 &nbsp;
                                <span style="background: rgba(139, 92, 246, 0.2); color: #C4B5FD; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 8px;">Grade {grade}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_v:
                        if st.button("View Analysis", key=f"hist_v_{item.get('id', idx)}", use_container_width=True):
                            load_analysis_into_session(item)
                            st.session_state.current_page = "📊 Analysis"
                            st.rerun()
                    with col_d:
                        if item.get("pdf"):
                            st.download_button(
                                "Download Report",
                                data=item["pdf"],
                                file_name=clean_fn,
                                mime="application/pdf",
                                key=f"hist_d_{item.get('id', idx)}",
                                use_container_width=True
                            )
                    with col_del:
                        if st.button("Delete", key=f"hist_del_{item.get('id', idx)}", use_container_width=True):
                            st.session_state[f"confirm_del_{item.get('id', idx)}"] = True
                            st.rerun()
                            
                    # Confirmation Step to Prevent Accidental Deletion
                    if st.session_state.get(f"confirm_del_{item.get('id', idx)}", False):
                        st.markdown(
                            """
                            <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 8px; padding: 10px 14px; margin-top: 8px;'>
                                <span style='color: #F87171; font-weight: 700; font-size: 13px;'>Delete Analysis?</span>
                                <div style='color: #CBD5E1; font-size: 12px; margin-top: 2px;'>This will remove this analysis record and its generated report snapshot.</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        cd_c1, cd_c2, _ = st.columns([1, 1.2, 4])
                        with cd_c1:
                            if st.button("Cancel", key=f"cancel_del_{item.get('id', idx)}"):
                                st.session_state[f"confirm_del_{item.get('id', idx)}"] = False
                                st.rerun()
                        with cd_c2:
                            if st.button("Delete", type="primary", key=f"confirm_del_btn_{item.get('id', idx)}"):
                                ok, msg = auth.delete_analysis_record(item.get("id"), username)
                                if ok:
                                    st.session_state.analysis_history[username] = [
                                        h for h in st.session_state.analysis_history[username] if h.get("id") != item.get("id")
                                    ]
                                    st.session_state[f"confirm_del_{item.get('id', idx)}"] = False
                                    st.success("✓ Analysis deleted successfully.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete analysis: {msg}")

# ----------------------------------------------------
# PAGE 8: SETTINGS
# ----------------------------------------------------
elif st.session_state.current_page in ["⚙ Settings", "⚙️ Settings"]:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">Settings & Workspace</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">Manage your account details, workspace preferences, and application data.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Account Profile")
    st.markdown(
        f"""
        <div class="iq-card">
            <p><b>Username:</b> {username}</p>
            <p><b>Academic Role:</b> {role}</p>
            <p><b>Total Analyses in Database:</b> {len(user_history)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Preferences")
    
    current_pref = st.session_state.get("theme_preference", "System")
    if current_pref not in ["Light", "Dark", "System"]:
        current_pref = "System"
    eff_theme = get_effective_theme(current_pref)
    
    theme_card_html = f"""
    <div class="iq-card" style="margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div>
                <div style="font-weight: 700; font-size: 15px;">Interface Theme</div>
                <div style="font-size: 13px; opacity: 0.8; margin-top: 2px;">Customize application workspace styling across all views</div>
            </div>
            <span style="background: rgba(139, 92, 246, 0.2); color: #8B5CF6; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.35);">
                Active: {current_pref} {f"({eff_theme} Mode)" if current_pref == "System" else "Mode"}
            </span>
        </div>
    </div>
    """
    st.markdown(theme_card_html, unsafe_allow_html=True)
    
    theme_options = ["Light", "Dark", "System"]
    theme_idx = theme_options.index(current_pref)
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        selected_theme = st.radio(
            "Appearance Mode",
            options=theme_options,
            index=theme_idx,
            horizontal=True,
            key="theme_mode_radio",
            help="Select Light for daytime reading, Dark for nighttime focus, or System to follow your OS setting automatically."
        )
    with col_t2:
        desc_map = {
            "Light": "Clean, crisp light interface with dark typography.",
            "Dark": "Signature dark navy & purple analytical workspace.",
            "System": f"Automatically tracks your operating system appearance (currently {eff_theme})."
        }
        st.markdown(
            f"""
            <div style="padding: 10px 14px; background: rgba(139, 92, 246, 0.08); border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.2); margin-top: 6px;">
                <div style="font-size: 11px; font-weight: 700; opacity: 0.8; text-transform: uppercase;">Selected Mode</div>
                <div style="font-size: 14px; font-weight: 700; color: #8B5CF6; margin-top: 2px;">{selected_theme}</div>
                <div style="font-size: 11px; opacity: 0.75; margin-top: 2px;">{desc_map.get(selected_theme, "")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    if selected_theme != current_pref:
        st.session_state["theme_preference"] = selected_theme
        if username:
            auth.update_user_theme(username, selected_theme)
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Data Management")
    if st.session_state.get("history_cleared_msg", False):
        st.success("✓ Analysis history cleared successfully.")
        st.session_state["history_cleared_msg"] = False
        
    st.markdown(
        f"""
        <div class="iq-card">
            <p><b>Current Storage:</b> {len(user_history)} archived paper evaluation records.</p>
            <p style="color: #94A3B8; font-size: 13px;">Clearing analysis history will permanently remove all archived score records and PDF snapshots for account <b>{username}</b>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🗑️ Clear All Analysis History", key="btn_clear_all_hist"):
        st.session_state["confirm_clear_all"] = True
        st.rerun()
        
    if st.session_state.get("confirm_clear_all", False):
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
                <div style="color: #F87171; font-weight: 700; font-size: 14px; margin-bottom: 6px;">⚠ Clear Analysis History</div>
                <div style="color: #CBD5E1; font-size: 13px; line-height: 1.6;">
                    This will permanently delete:
                    <ul style="margin: 6px 0 0 18px; padding: 0;">
                        <li>Analysis records</li>
                        <li>Stored scores</li>
                        <li>Generated report snapshots</li>
                        <li>Related history metadata</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        c_cancel, c_confirm, _ = st.columns([1, 1.6, 4])
        with c_cancel:
            if st.button("Cancel", key="btn_cancel_clear_all"):
                st.session_state["confirm_clear_all"] = False
                st.rerun()
        with c_confirm:
            if st.button("Delete Everything", type="primary", key="btn_confirm_clear_all_exec"):
                success, msg = auth.clear_user_history(username)
                if success:
                    st.session_state.analysis_history[username] = []
                    st.session_state["confirm_clear_all"] = False
                    st.session_state["history_cleared_msg"] = True
                    st.rerun()
                else:
                    st.error(f"Failed to clear history: {msg}")
                
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Privacy & Local Processing")
    st.markdown(
        """
        <div class="iq-card">
            <p><b>Local In-Memory Execution:</b> All text extraction (PDF, DOCX, TXT), statistical natural language processing, and local database operations execute entirely within your local environment.</p>
            <p><b>Local Database Isolation:</b> Analysis records, scores, and generated PDF reports are persisted exclusively inside your local SQLite database.</p>
            <p><b>External AI Processing:</b> PaperIQ utilizes deterministic computational linguistics and natural language processing routines. No paper text or extracted data is transmitted to external model providers or training datasets.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------
# PAGE 9: ABOUT & METHODOLOGY
# ----------------------------------------------------
elif st.session_state.current_page in ["ℹ About", "ℹ️ About"]:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">About PaperIQ</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 15px;">AI-Powered Research Paper Analysis & Evaluation System.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="iq-card">
            <h4>What is PaperIQ?</h4>
            <p>PaperIQ is an intelligent academic paper evaluation platform that analyzes research manuscripts for language quality, structural coherence, argumentation density, academic style, and readability. It empowers students, researchers, faculty, and peer evaluators to identify structural gaps, understand manuscript writing quality, and generate comprehensive review reports.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="iq-card">
            <h4>Analytical Scoring Framework</h4>
            <p>PaperIQ evaluates papers across five weighted quality dimensions to calculate a transparent composite score (0 – 100):</p>
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #1F2937; border-radius: 8px; padding: 14px 18px; margin: 12px 0 16px 0; font-size: 13px; color: #A78BFA; line-height: 1.8;">
                <b>Overall Writing Score =</b><br>
                &nbsp;&nbsp;Language Quality × 25%<br>
                + Structural Coherence × 25%<br>
                + Argumentation × 20%<br>
                + Academic Style × 15%<br>
                + Readability × 15%
            </div>
            <ul>
                <li><b>Language Quality (25%):</b> Evaluates syntactic cadence, sentence length calibration, and formal academic phrasing.</li>
                <li><b>Structural Coherence (25%):</b> Measures the density and placement of transitional discourse markers between paragraphs and sections.</li>
                <li><b>Argumentation (20%):</b> Evaluates causal links and evidentiary propositions supporting analytical assertions.</li>
                <li><b>Academic Style (15%):</b> Measures lexical richness, technical terminology, and scholarly vocabulary diversity.</li>
                <li><b>Readability (15%):</b> Evaluates reading ease calibrated for academic literature and peer-reviewed manuscripts.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="iq-card">
            <h4>Responsible AI Statement & System Limitations</h4>
            <p>PaperIQ evaluates textual, linguistic, and structural characteristics of scholarly manuscripts. It does not independently verify scientific correctness, empirical veracity, experimental reproducibility, novelty, or journal acceptance. A manuscript's intrinsic intellectual contribution cannot be captured solely by computational linguistic metrics. All AI-generated feedback and suggestions should be treated as assistance, not final academic judgment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="iq-card">
            <h4>Technology Stack Architecture</h4>
            <p><b>Application Core:</b> Python, Streamlit</p>
            <p><b>NLP & Computational Linguistics:</b> TextBlob, Scikit-learn (TF-IDF & Cosine Similarity), NLTK</p>
            <p><b>Document Processing:</b> pdfplumber, python-docx, FPDF</p>
            <p><b>Data Visualization:</b> Plotly Graph Objects (Radar & Comparative Charts)</p>
            <p><b>Storage & Security:</b> Local SQLite database, SHA-256 password hashing</p>
        </div>
        """,
        unsafe_allow_html=True
    )


