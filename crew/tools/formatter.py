# FILE: crew/tools/formatter.py
# Purpose: Consistent markdown card + LinkedIn post text.

from datetime import date

def news_card_md(title, cleaned_summary, bullets, image_path, link):
    """
    Returns a markdown 'news card' with date, summary, bullets, and an image placeholder.
    """
    d = date.today().isoformat()
    bullets_md = "\n".join([f"- {b}" for b in bullets])
    return f"""# {title}
_Date: {d}_

{cleaned_summary}

**Key Takeaways**
{bullets_md}

![illustration]({image_path})

Source: {link}
"""

def linkedin_post(title, one_liner, tags):
    """
    Returns a short LinkedIn post (2–3 sentences + hashtags).
    """
    tags_str = " ".join(f"#{t}" for t in tags)
    return f"""📰 {title}

{one_liner}

{tags_str}
"""
