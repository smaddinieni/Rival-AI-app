# langgraph_pipeline/utils.py
from datetime import datetime
import re


def format_timestamp(iso_timestamp: str) -> str:
    """Converts an ISO format timestamp string to a more readable format."""
    try:
        dt_object = datetime.fromisoformat(iso_timestamp)
        # Example format: May 05, 2025 03:15 AM
        return dt_object.strftime("%b %d, %Y %I:%M %p")
    except (ValueError, TypeError):
        return (
            "Invalid Date"  # Handle cases where timestamp might be missing or invalid
        )


def calculate_word_count(text: str) -> int:
    """Calculates the approximate word count of a given text."""
    if not text or not isinstance(text, str):
        return 0
    # Simple split by whitespace, might not be perfect for all cases
    words = text.split()
    return len(words)


def sanitize_filename(name: str, default: str = "download") -> str:
    """Removes or replaces characters unsuitable for filenames."""
    # Remove characters that are not alphanumeric, underscore, or hyphen
    sanitized = re.sub(r"[^\w\-]+", "_", name)
    # Remove leading/trailing underscores/hyphens and reduce multiple consecutive ones
    sanitized = re.sub(r"^[_ \-]+|[_ \-]+$", "", sanitized)
    sanitized = re.sub(r"[_ \-]+", "_", sanitized)
    # Return default if the name becomes empty after sanitization
    return sanitized if sanitized else default


# --- Placeholder for more complex analysis ---
# You could add functions here to call NLP libraries (like spaCy, NLTK)
# or even another LLM call to analyze tone, sentiment, length comparison, etc.
# Example:
# def analyze_qualitative_deltas(text1: str, text2: str) -> dict:
#     """
#     Placeholder for analyzing differences between two texts.
#     In a real implementation, this might involve NLP libraries or LLM calls.
#     """
#     len1 = calculate_word_count(text1)
#     len2 = calculate_word_count(text2)
#     delta = {
#         "length_diff": len2 - len1,
#         "tone_comparison": "Gemini tone seems more empathetic (Mock Analysis)",
#         "sentiment_comparison": "Gemini sentiment slightly more positive (Mock Analysis)"
#     }
#     return delta

# --- Placeholder for Export Functions ---
# Example:
# def export_to_pdf(content: str, filename: str):
#     """Placeholder for exporting text content to a PDF file."""
#     # This would require a library like reportlab or fpdf
#     print(f"Placeholder: Exporting content to {filename}.pdf")
#     # try:
#     #     from reportlab.pdfgen import canvas
#     #     from reportlab.lib.pagesizes import letter
#     #     c = canvas.Canvas(f"{filename}.pdf", pagesize=letter)
#     #     textobject = c.beginText(40, 750)
#     #     for line in content.splitlines():
#     #         textobject.textLine(line)
#     #     c.drawText(textobject)
#     #     c.save()
#     #     return True
#     # except ImportError:
#     #     print("ReportLab not installed. Cannot export to PDF.")
#     #     return False
#     # except Exception as e:
#     #     print(f"Error exporting to PDF: {e}")
#     #     return False
#     pass
