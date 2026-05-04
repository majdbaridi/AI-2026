Flow of file 1 React Loop Agent


User question
   ↓
LangChain messages
   ↓
LLM with bound tools
   ↓
LLM decides tool call
   ↓
Python executes tool
   ↓
Tool result added back as ToolMessage
   ↓
LLM continues
   ↓
Final answer


File 2 just a draft wiht no langchain:
                  User
                   |
                   v
        "What is laptop price with gold discount?"
                   |
                   v
              Agent Loop
                   |
                   v
        LLM receives question + tool list
                   |
                   v
        LLM chooses next action
          /                     \
         /                       \
        v                         v
get_product_price("laptop")   apply_discount(...)
        |                         |
        v                         v
     1299.99                  1000.99
        \                         /
         \                       /
          v                     v
        Tool result added to messages
                   |
                   v
          LLM produces final answer



conclusion:
## 🧠 ReAct Pattern in This Project

This project implements a **ReAct-style agent (Reasoning + Acting)** using modern tool-calling instead of explicit “Thought/Action” prompts.

---

### Reasoning

```python
ai_message = llm_with_tools.invoke(messages)

This is where reasoning happens inside the LLM.

The model:

Reads the system prompt
Understands available tools
Decides:
Should I answer directly?
Or call a tool?
Which tool?
With what arguments?

👉 This decision-making step = Reasoning

⚡ Acting
tool_call = ai_message.tool_calls[0]
tool_name = tool_call.get("name")
tool_args = tool_call.get("args", {})

observation = tool_to_use.invoke(tool_args)

👉 This is Acting

The LLM selects a tool
The application executes it
👀 Observation
messages.append(
    ToolMessage(
        content=str(observation),
        tool_call_id=tool_call_id,
    )
)

👉 This is Observation

The tool result is sent back to the LLM
The model uses it for the next reasoning step
🔁 Full ReAct Loop
1. Reasoning → llm_with_tools.invoke()
2. Acting → execute tool
3. Observation → add ToolMessage
4. Repeat
5. Final Answer
📌 Where is the ReAct Prompt?

It is embedded in the SystemMessage:

SystemMessage(content=(
    "You are a helpful shopping assistant.\n"
    ...
    "STRICT RULES:\n"
    "1. If the user asks about a product price, you MUST call get_product_price first.\n"
    ...
))

This defines:

Tool usage rules
Execution constraints
Agent behavior

This is a production-style ReAct implementation where reasoning is implicit and actions are executed via structured tool calls.
