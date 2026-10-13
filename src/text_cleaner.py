import re
import html
import unicodedata

def clean_text(text: str) -> str:
    """
    Cleans and standardizes raw text from Resumes or Job Descriptions.
    Preserves programming keywords like C++, C#, and .NET while cleaning noise.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # 2. Normalize unicode characters (turns weird bullets/dashes into standard characters)
    text = unicodedata.normalize("NFKD", text)

    # 3. Remove web URLs (http://..., https://..., www....)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)

    # 5. Remove phone numbers (various international and standard formats)
    text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)

    # 6. Remove non-printable characters and bullet artifacts (\xa0, \u2022, etc.)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 7. Clean special characters BUT preserve tech tokens like C++, C#, .NET
    # Replace symbols except +, #, ., and standard punctuation
    text = re.sub(r'[^\w\s\+\#\.\,\-\/\:]', ' ', text)

    # 8. Collapse multiple spaces and newlines into single spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


if __name__ == "__main__":
    # Quick sanity test
    sample_dirty_text = """
    John Doe | Software Engineer
    Email: john.doe@email.com | Phone: +1 (555) 123-4567 | Website: https://johndoe.dev
    • Proficient in C++, C#, and .NET core architecture.
    • Experience with &amp; Python 3.9, SQL, and AWS.
    \xa0\xa0\xa0\xa0Excessive whitespace and weird symbols: *** === >>>
    """

    print("=" * 60)
    print("🧹 TESTING TEXT CLEANER")
    print("=" * 60)
    print("RAW INPUT:\n", sample_dirty_text)
    print("-" * 60)
    cleaned = clean_text(sample_dirty_text)
    print("CLEANED OUTPUT:\n", cleaned)
    print("=" * 60)