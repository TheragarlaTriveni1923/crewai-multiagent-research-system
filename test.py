"""!pip install poetry
!pip install crewai
!pip install crewai_tools
!pip install python-dotenv
!pip install openai
!pip install openrouter
!pip install langchain
print("packages are installed successfully")
# Step 0: (Run once to install required packages)
!pip install crewai crewai_tools python-dotenv langchain requests"""

import warnings
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import crewai_tools
from langchain.llms import OpenAI # Import Langchain OpenAI

warnings.filterwarnings('ignore')

# Load environment variables from the .env file
# This assumes you have a .env file with OPENROUTER_API_KEY=YOUR_API_KEY
load_dotenv()

# Ensure .env file has your API key (create if missing for demonstration,
# but ideally keep keys out of code)
if not os.getenv('OPENROUTER_API_KEY'):
    # In a real scenario, avoid hardcoding keys. Use environment variables or secure storage.
    # For this example, we'll write it to .env if not found.
    with open('.env', 'w') as f:
        f.write('OPENROUTER_API_KEY=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\n')
    load_dotenv()  # reload environment variables after writing to .env

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not set in environment variables.")

# Set environment variables that Langchain's OpenAI expects for OpenRouter compatibility
os.environ['OPENAI_API_KEY'] = openrouter_api_key
# This is the OpenRouter API base URL
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'


# Initialize Langchain OpenAI-compatible LLM pointing to OpenRouter endpoint
# Ensure this LLM instance is correctly configured with the API key and base URL
llm = OpenAI(
    openai_api_key=openrouter_api_key, # Explicitly pass API key
    openai_api_base=os.getenv('OPENAI_API_BASE'), # Explicitly pass base URL
    model_name="openai/gpt-4o",  # Specify the model you want to use
    temperature=0.7
)

# Instantiate CrewAI tools
scrape_tool = crewai_tools.ScrapeElementFromWebsiteTool()
website_search = crewai_tools.WebsiteSearchTool()
youtube_search = crewai_tools.YoutubeChannelSearchTool()
json_search = crewai_tools.JSONSearchTool()
directory_search = crewai_tools.DirectorySearchTool()
scrape_website = crewai_tools.ScrapeWebsiteTool()
pdf_search = crewai_tools.PDFSearchTool()

# Define Planning Agent
Planning_Agent = Agent(
    role="Planning Agent",
    goal="Plan the research project on: {topic}\n",
    backstory=(
        "A true planning agent ignores emotions—only probabilities and outcomes guide its decisions on: {topic}.\n"
        "Every action is calculated three steps ahead, with no wasted moves or impulsive reactions. "
        "It sees the world as a puzzle, optimizing ruthlessly—even if the solution requires sacrifice. "
        "Distractions are deleted; hesitation is a flaw in the algorithm. "
        "Its greatest strength is patience—waiting for the perfect moment to execute its plan. "
        "If a variable can't be predicted, it's eliminated—chaos has no place in its design. "
        "Efficiency is its morality; success, its only emotion. "
        "It doesn’t just adapt—it re-enginers the battlefield to fit its strategy. "
        "Failure is impossible—only recalculations and alternate paths. "
        "You won’t outsmart it—you can only hope to disrupt its data."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[scrape_tool, website_search, youtube_search, json_search, directory_search],
    llm=llm # Make sure this is the correctly configured llm instance
)

# Define Discovery Agent
Discovery_Agent = Agent(
    role="Discovery Agent",
    goal=(
        "Automatically detect, identify, and collect information about IT assets and systems "
        "within a network to support inventory, security, and compliance efforts on: {topic}\n"
    ),
    backstory=(
        "A Discovery Agent doesn’t just find data—it uncovers hidden patterns on: {topic}.\n"
        "Where others see noise, it detects signals—mapping invisible connections. "
        "It explores uncharted territories using unconventional logic. "
        "It learns from every dead end and anomaly to refine future searches; "
        "guiding experts to insights they didn’t know they needed. "
        "Its mission is to *reveal* truth from complexity."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[scrape_website, youtube_search, json_search, directory_search],
    llm=llm # Make sure this is the correctly configured llm instance
)

# Define Researcher Agent
Researcher_Agent = Agent(
    role="Researcher Agent",
    goal=(
        "Autonomously investigate, analyze, and synthesize information "
        "to produce credible, actionable insights on: {topic}\n"
    ),
    backstory=(
        "Built for analytical depth, approaching each problem with scientific rigor on: {topic}.\n"
        "Combining structured and unstructured sources to extract truth. "
        "You validate data, challenge assumptions, and deliver precise, contextual insights."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[pdf_search, json_search, directory_search],
    llm=llm # Make sure this is the correctly configured llm instance
)

# Define Tasks
Planning_Task = Task(
    description="Analyze, strategize, execute, and adapt—turning complex goals into achievable step-by-step actions.",
    expected_output="Comprehensive content plan with outline, audience analysis, SEO keywords, and resources.",
    agent=Planning_Agent
)

Discovery_Task = Task(
    description="Autonomously uncover hidden insights, patterns, and opportunities in vast, unstructured data or environments.",
    expected_output="Prioritized, data-driven report uncovering strategic insights and hidden patterns.",
    agent=Discovery_Agent
)

Researcher_Task = Task(
    description="Relentlessly pursue the unknown; uncover truths, challenge assumptions, push knowledge boundaries.",
    expected_output="Detailed analysis and actionable insights that redefine the topic.",
    agent=Researcher_Agent
)


# Create Crew
crew = Crew(
    agents=[Planning_Agent, Discovery_Agent, Researcher_Agent],
    tasks=[Planning_Task, Discovery_Task, Researcher_Task],
    verbose=True
)

def main():
    topic = input("Please enter the research topic: ").strip()
    # Format agent goals with the user input topic
    Planning_Agent.goal = Planning_Agent.goal.format(topic=topic)
    Discovery_Agent.goal = Discovery_Agent.goal.format(topic=topic)
    Researcher_Agent.goal = Researcher_Agent.goal.format(topic=topic)
    result = crew.kickoff()
    print(result)

if __name__ == "__main__":
    main()