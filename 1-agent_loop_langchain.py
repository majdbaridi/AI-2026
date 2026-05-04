from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """
    Fetches the price of a product from the demo catalog.

    Available products:
    - laptop
    - headphones
    - keyboard
    """
    print(f">> Executing get_product_price(product='{product}')")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    return prices.get(product.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Applies a discount tier to a price and returns the final price.

    Available tiers:
    - bronze: 5%
    - silver: 12%
    - gold: 23%
    """
    print(
        f">> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant.\n"
                "You have access to two tools:\n"
                "1. get_product_price: gets product prices from the demo catalog.\n"
                "2. apply_discount: applies a discount to a price.\n\n"
                "Available products in the catalog are: laptop, headphones, keyboard.\n"
                "Available discount tiers are: bronze, silver, gold.\n\n"
                "STRICT RULES:\n"
                "1. If the user asks about a product price, you MUST call get_product_price first.\n"
                "2. Use the exact product names: laptop, headphones, keyboard.\n"
                "3. If the product is not in the catalog, say that the product is not available.\n"
                "4. Only call apply_discount AFTER get_product_price returns a valid price.\n"
                "5. Pass the exact price returned by get_product_price to apply_discount.\n"
                "6. NEVER calculate discounts yourself. Always use apply_discount.\n"
                "7. If the user asks for a discount but does not specify the tier, ask which tier to use.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id,
            )
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()

    user_input = input("Ask your question: ")
    run_agent(user_input)