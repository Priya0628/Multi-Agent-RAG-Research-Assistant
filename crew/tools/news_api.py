# FILE: crew/tools/news_api.py
# Purpose: Fetch a top story from an RSS feed as our "article" source.

import os
import feedparser #library to read the RSS URL

def fetch_top_story():
    """
    Reads NEWS_RSS_URL from environment and returns a dict with:
    { "title": str, "summary": str, "link": str }
    """
    rss = os.getenv("NEWS_RSS_URL", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml") #Public RSS feed
    feed = feedparser.parse(rss)
    if not feed.entries:
        return {"title": "(no items)", "summary": "", "link": ""}

    e = feed.entries[0]
    title = getattr(e, "title", "")
    summary = getattr(e, "summary", getattr(e, "description", ""))
    link = getattr(e, "link", "")
    return {"title": title, "summary": summary, "link": link}
