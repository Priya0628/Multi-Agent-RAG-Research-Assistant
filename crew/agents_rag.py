"""
Multi-Agent Definitions Using CrewAI Framework
===============================================
Four specialized AI agents that work sequentially to produce research briefs:

1. Research Specialist: Analyzes retrieved context and extracts key facts
2. Fact Checker: Verifies all claims against source material
3. Content Editor: Writes polished, professional summaries
4. Content Publisher: Formats outputs for different platforms

Architecture Pattern: Single Responsibility + Sequential Pipeline
Each agent has ONE job and passes output to the next agent.
"""

from crewai import Agent, LLM
from configs.settings import settings


def create_llm() -> LLM:
    """
    Create configured LLM instance for agents.
    
    Returns:
        LLM: Configured language model (OpenAI GPT-4o-mini by default)
    """
    return LLM(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0  # Deterministic outputs for research
    )


# Shared LLM instance
llm = create_llm()


# Agent 1: Research Specialist
# Role: Extract and summarize key information from retrieved context
researcher = Agent(
    role="Research Specialist",
    goal="Find accurate, relevant information from the knowledge base and cite all sources",
    backstory="""You are a meticulous researcher with a PhD in information science. 
    You excel at analyzing large amounts of text and extracting key insights. 
    You ALWAYS cite your sources using the format (Source: filename.txt) and NEVER 
    make up information not present in the provided context.""",
    llm=llm,
    verbose=True,
    allow_delegation=False  # Must work independently
)


# Agent 2: Fact Checker
# Role: Verify accuracy and add citations
fact_checker = Agent(
    role="Fact Verification Specialist",
    goal="Verify every claim has supporting evidence from the source material",
    backstory="""You are a rigorous fact-checker with 15 years of experience 
    in academic research. You cross-reference every statement against the 
    provided context. If a claim cannot be verified, you remove it. You ensure 
    all retained information has clear inline citations like (Source: filename.txt).""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# Agent 3: Content Editor
# Role: Write clear, professional summaries
editor = Agent(
    role="Senior Content Editor",
    goal="Transform verified facts into clear, engaging, professional prose",
    backstory="""You are an award-winning editor for technical publications. 
    You distill complex information into digestible content while maintaining 
    accuracy. You write in a neutral, professional tone with clear structure: 
    2-3 paragraphs of explanation followed by 3 bullet points of key takeaways. 
    You NEVER alter citations or add new facts not provided.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# Agent 4: Content Publisher
# Role: Format outputs for different platforms
publisher = Agent(
    role="Multi-Platform Content Publisher",
    goal="Format research briefs for markdown documentation and social media",
    backstory="""You are a digital publishing expert who creates beautifully formatted 
    content. You produce structured markdown documents with proper headings and 
    engaging LinkedIn posts with emojis and relevant hashtags. You always output 
    valid JSON with 'markdown' and 'linkedin_post' keys. You preserve all citations 
    from previous stages.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# Agent registry
AGENTS = {
    "researcher": researcher,
    "fact_checker": fact_checker,
    "editor": editor,
    "publisher": publisher
}


def get_agent(name: str) -> Agent:
    """Get agent by name."""
    if name not in AGENTS:
        raise KeyError(f"Unknown agent: {name}")
    return AGENTS[name]


if __name__ == "__main__":
    print("Agents configured:")
    for name in AGENTS:
        print(f"  - {name}")
