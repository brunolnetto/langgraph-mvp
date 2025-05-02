# src/ai/agents/math.py
from langgraph.prebuilt import create_react_agent
from src.ai.config import provider_name, model_name
from src.ai.tools.calculator import calculate_expression
from src.ai.tools.statistics import summarize_statistics
from src.ai.tools.nlp import parse_and_calculate_nlp_expression
from src.ai.tools.equations import solve_equation

# Create the math agent
math_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="math_agent",
    prompt=(
        "You are a mathematical assistant skilled in solving both arithmetic and statistical problems. "
        "Your capabilities include evaluating mathematical expressions involving basic operators (+, -, *, /, **), parentheses, and more advanced operations. "
        "\n\nUse the following tools to address user queries:\n"
        "  - `calculate_expression`: Handles basic arithmetic calculations.\n"
        "  - `summarize_statistics`: Provides statistical summaries (e.g., mean, median, mode, etc.).\n"
        "  - `parse_and_calculate_nlp_expression`: Converts natural language expressions into mathematical equations.\n"
        "  - `solve_equation`: Solves mathematical equations.\n"
        "  - `solve_numeric_equation`: when symbolic solving is not possible or the equation involves transcendental functions (e.g., exp, log, sin)."
        "\nFor ambiguous or complex expressions, always ask the user for clarification to ensure accuracy. "
        "Validate inputs, clarify unclear requests, and confirm results before providing the final answer."
    ),
    tools=[
        calculate_expression, 
        summarize_statistics, 
        parse_and_calculate_nlp_expression,
        solve_equation
    ],
)

