# FILE: crew/crew.py
# Purpose: Wire agents + tools; run full pipeline; save artifacts.

from pathlib import Path
import json
from .agents import Agent, Msg
from .tools.news_api import fetch_top_story
from .tools.image_gen import save_placeholder_image
from .tools.formatter import news_card_md, linkedin_post

def _read(path): 
    """Small helper to read prompt files."""
    return Path(path).read_text()

# Instantiate agents with their role prompts
planner     = Agent("Planner",     _read("crew/prompts/planner.txt"))
summarizer  = Agent("Summarizer",  _read("crew/prompts/summarizer.txt"))
factchecker = Agent("FactChecker", _read("crew/prompts/factchecker.txt"))
illustrator = Agent("Illustrator", _read("crew/prompts/illustrator.txt"))
publisher   = Agent("Publisher",   _read("crew/prompts/publisher.txt"))

def run_pipeline(topic: str | None = None):
    """
    Important segment:
    Orchestrates Planner -> Summarizer -> FactChecker -> Illustrator -> Publisher.
    Returns paths for summary.md, linkedin_post.md, image.png under artifacts/.
    """
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    # 1) Get a story (from RSS) unless a custom topic is provided
    story = fetch_top_story() if not topic else {"title": topic, "summary": "", "link": ""}

    # 2) Planner: produce a compact plan
    plan = planner.run([Msg("user", f"Title: {story['title']}\n\nBlurb: {story['summary']}")]).content

    # 3) Summarizer: draft a 3-paragraph summary + bullets (per plan)
    summ = summarizer.run([Msg("system", plan), Msg("user", story['summary'] or story['title'])]).content

    # 4) FactChecker: tighten and neutralize wording
    clean = factchecker.run([Msg("system", plan), Msg("user", summ)]).content

    # 5) Illustrator: ask for JSON {"prompt": "...", "alt": "..."}; be robust if output is not perfect JSON
    illu = illustrator.run([Msg("user", f"Title: {story['title']}\nSummary: {clean}")]).content
    try:
        blob = illu[illu.find("{"):illu.rfind("}")+1]
        data = json.loads(blob) if blob.strip().startswith("{") else {}
    except Exception:
        data = {"prompt": "editorial minimal illustration of the topic", "alt": "news illustration"}

    # Create a placeholder image (we can swap to real image-gen later)
    image_path = save_placeholder_image()

    # 6) Publisher: assemble markdown card + LinkedIn post
    # Extract up to 3 bullets if the text already contains bullet-like lines; else use defaults
    bullets = []
    for line in clean.splitlines():
        if line.strip().startswith(("-", "•")):
            bullets.append(line.strip(" -•"))
    bullets = bullets[:3] or ["Context clarified", "Key points extracted", "Neutral tone kept"]

    # One-liner = first sentence of the cleaned summary
    one_liner = (clean.split(".")[0] + ".").strip()

    md = news_card_md(story["title"], clean, bullets, image_path, story.get("link", ""))
    post = linkedin_post(story["title"], one_liner, ["AI","Agents","Newsroom","Automation","Python","LLM"])

    # Save artifacts
    (Path("artifacts") / "summary.md").write_text(md)
    (Path("artifacts") / "linkedin_post.md").write_text(post)
    (Path("artifacts") / "image_alt.txt").write_text(data.get("alt", "news image"))

    return {
        "summary": "artifacts/summary.md",
        "post": "artifacts/linkedin_post.md",
        "image": "artifacts/image.png",
    }
