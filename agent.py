"""
AutoStream Conversational AI Agent
====================================
Stack  : LangGraph (StateGraph) + Gemini 2.5 Flash
Pattern: Social-to-Lead Agentic Workflow
Company: ServiceHive / Inflx — ML Intern Assignment
"""

import json
import os
import re
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

# ── LangGraph ──────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ── LangChain ──────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


# 1. KNOWLEDGE BASE (RAG)

def load_knowledge_base(path: str = "knowledge_base.json") -> str:
    with open(path, "r") as f:
        kb = json.load(f)

    basic = kb["plans"]["basic"]
    pro   = kb["plans"]["pro"]
    ent   = kb["plans"]["enterprise"]
    pol   = kb["policies"]

    text = f"""
=== AutoStream Knowledge Base ===
Product : {kb['product_name']} — {kb['tagline']}

Basic Plan  ({basic['price']} | Annual: {basic['price_annual']})
  Videos: {basic['videos_per_month']}/month | Max: {basic['max_video_length']} | {basic['resolution']} | {basic['storage']}
  Includes: {', '.join(basic['features'])}
  Not included: {', '.join(basic['not_included'])}

Pro Plan  ({pro['price']} | Annual: {pro['price_annual']})
  Videos: {pro['videos_per_month']} | Max: {pro['max_video_length']} | {pro['resolution']} | {pro['storage']}
  Includes: {', '.join(pro['features'])}
  Not included: {', '.join(pro['not_included'])}

Enterprise Plan  ({ent['price']})
  Includes: {', '.join(ent['features'])}

POLICIES:
Refund: {pol['refund']}
Free Trial: {pol['free_trial']}
Support Basic: {pol['support']['basic']}
Support Pro: {pol['support']['pro']}
Cancellation: {pol['cancellation']}
Annual Discount: {pol['annual_discount']}

FAQ:"""
    for faq in kb["faq"]:
        text += f"\nQ: {faq['question']}\nA: {faq['answer']}\n"

    return text.strip()


# 2. TOOL — Mock Lead Capture

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Capture a qualified lead.
    Call ONLY when name, email, AND platform are all collected.

    Args:
        name     : Full name of the lead
        email    : Email address of the lead
        platform : Creator platform (YouTube, Instagram, TikTok, etc.)
    """
    print("\n" + "=" * 50)
    print("LEAD CAPTURED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print("=" * 50 + "\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# 3. STATE DEFINITION

class AgentState(TypedDict):
    messages        : Annotated[list, add_messages]
    intent          : str
    lead_name       : Optional[str]
    lead_email      : Optional[str]
    lead_platform   : Optional[str]
    collecting_lead : bool
    lead_captured   : bool


# 4. NODES
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError(
        "\n❌ GOOGLE_API_KEY environment variable not set!\n"
        "Get a free API key at: https://aistudio.google.com/app/apikey\n"
        "Then set it in PowerShell: $env:GOOGLE_API_KEY='your_api_key_here'\n"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key
)
llm_tools      = llm.bind_tools([mock_lead_capture])
KNOWLEDGE_BASE = load_knowledge_base("knowledge_base.json")


def classify_intent_node(state: AgentState) -> dict:
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    system = SystemMessage(content="""You are an intent classifier for a SaaS sales agent.
Classify the user message into EXACTLY one label:
  casual_greeting   — greetings, thanks, small talk
  product_inquiry   — questions about features, pricing, plans, policies
  high_intent_lead  — user wants to sign up, try, purchase, or subscribe
Reply with ONLY the label, nothing else.""")

    response = llm.invoke([system, HumanMessage(content=last_user_msg)])
    raw = response.content.strip().lower()

    if "greeting" in raw:
        intent = "casual_greeting"
    elif "high" in raw or "lead" in raw:
        intent = "high_intent_lead"
    else:
        intent = "product_inquiry"

    return {"intent": intent}


def extract_lead_node(state: AgentState) -> dict:
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    system = SystemMessage(content="""Extract lead info from the message.
Return ONLY valid JSON with keys: name, email, platform. Use null for missing.
No markdown, no extra text.""")

    response = llm.invoke([system, HumanMessage(content=last_user_msg)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = {"name": None, "email": None, "platform": None}

    updates: dict = {}
    if not state.get("lead_name") and extracted.get("name"):
        updates["lead_name"] = extracted["name"]
    if not state.get("lead_email") and extracted.get("email"):
        updates["lead_email"] = extracted["email"]
    if not state.get("lead_platform") and extracted.get("platform"):
        updates["lead_platform"] = extracted["platform"]

    return updates


def respond_node(state: AgentState) -> dict:
    missing = []
    if not state.get("lead_name"):     missing.append("full name")
    if not state.get("lead_email"):    missing.append("email address")
    if not state.get("lead_platform"): missing.append("creator platform (YouTube, Instagram, etc.)")

    lead_instruction = ""
    if state.get("collecting_lead") and not state.get("lead_captured"):
        if missing:
            lead_instruction = (
                f"\n\nACTIVE TASK: Lead collection in progress. "
                f"Still need: {', '.join(missing)}. "
                f"Ask for ONE missing field naturally. Do NOT call mock_lead_capture yet."
            )
        else:
            lead_instruction = (
                f"\n\nACTIVE TASK: All fields collected — "
                f"Name={state['lead_name']}, Email={state['lead_email']}, Platform={state['lead_platform']}. "
                f"Confirm warmly then call mock_lead_capture with these exact values."
            )

    system_prompt = f"""You are AutoStream's friendly AI sales assistant.
Answer using ONLY the knowledge base. Keep replies to 2-4 sentences.
{lead_instruction}

{KNOWLEDGE_BASE}"""

    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_tools.invoke(messages_to_send)

    new_messages  = [response]
    extra_updates : dict = {}

    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "mock_lead_capture":
                args   = tc["args"]
                result = mock_lead_capture.invoke(args)
                new_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                extra_updates["lead_captured"] = True

                confirm = llm.invoke([
                    SystemMessage(content="You are AutoStream's friendly AI assistant."),
                    HumanMessage(content=(
                        f"Lead captured for {args['name']} ({args['email']}, {args['platform']}). "
                        "Give a warm 1-2 sentence confirmation. Say the team will be in touch soon."
                    ))
                ])
                new_messages.append(confirm)

    return {"messages": new_messages, **extra_updates}


def start_lead_collection_node(state: AgentState) -> dict:
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    system = SystemMessage(content='Extract lead info. Return ONLY JSON: {"name": ..., "email": ..., "platform": ...}. Use null for missing.')
    response = llm.invoke([system, HumanMessage(content=last_user_msg)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        extracted = json.loads(raw)
    except Exception:
        extracted = {"name": None, "email": None, "platform": None}

    return {
        "collecting_lead" : True,
        "lead_name"       : extracted.get("name"),
        "lead_email"      : extracted.get("email"),
        "lead_platform"   : extracted.get("platform"),
    }


# 5. ROUTING

def route_after_intent(state: AgentState) -> Literal["start_lead_collection", "extract_lead", "respond"]:
    if state["intent"] == "high_intent_lead":
        if not state.get("collecting_lead"):
            return "start_lead_collection"
        return "extract_lead"
    return "respond"


# 6. BUILD GRAPH

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent",       classify_intent_node)
    builder.add_node("start_lead_collection", start_lead_collection_node)
    builder.add_node("extract_lead",          extract_lead_node)
    builder.add_node("respond",               respond_node)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "start_lead_collection": "start_lead_collection",
            "extract_lead"         : "extract_lead",
            "respond"              : "respond",
        },
    )
    builder.add_edge("start_lead_collection", "extract_lead")
    builder.add_edge("extract_lead",          "respond")
    builder.add_edge("respond",               END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ===== MAIN FIXED VERSION =====

def run_agent():

    graph = build_graph()
    config = {"configurable": {"thread_id": "autostream-session-1"}}

    print("\n" + "=" * 55)
    print("  AutoStream AI Sales Agent")
    print("  Powered by LangGraph + Gemini 2.5 Flash")
    print("=" * 55)
    print("  Type 'quit' to exit.\n")

    opener = ("Hi! Welcome to AutoStream — AI-powered video editing "
              "for content creators. How can I help you today?")
    print(f"Agent: {opener}\n")

    first_turn = True

    while True:
        user_input = input("You: ").strip()

        # 🔥 IMPORTANT FIX — empty input handle
        if user_input == "":
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nAgent: Thanks for chatting! Have a great day!\n")
            break

        if first_turn:
            invoke_input = {
                "messages": [
                    AIMessage(content=opener),
                    HumanMessage(content=user_input)
                ],
                "intent": "",
                "lead_name": None,
                "lead_email": None,
                "lead_platform": None,
                "collecting_lead": False,
                "lead_captured": False,
            }
            first_turn = False
        else:
            invoke_input = {
                "messages": [HumanMessage(content=user_input)]
            }

        result = graph.invoke(invoke_input, config=config)

        # 🔥 FIX — safe output printing
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\nAgent: {msg.content}\n")
                    break
        else:
            print("⚠️ No response from agent")


if __name__ == "__main__":
    run_agent()
# 7. MAIN LOOP

# def run_agent():
    
#     # ❌ ye commented hi rehne do
#     # if not os.environ.get("GOOGLE_API_KEY"):
#     #     raise EnvironmentError(...)

#     graph = build_graph()
#     config = {"configurable": {"thread_id": "autostream-session-1"}}

#     print("\n" + "=" * 55)
#     print(" AutoStream AI Sales Agent")
# #     if not os.environ.get("GOOGLE_API_KEY"):
# #         raise EnvironmentError(
# #             "GOOGLE_API_KEY not set!\n"
# #             "Get free key at: https://aistudio.google.com/app/apikey\n"
# #             "Then run in PowerShell: $env:GOOGLE_API_KEY='AIzaSyChzleCKoiDlM4-9TVNl7J0KOO1obF3s10'"
# #         

#     graph  = build_graph()
#     config = {"configurable": {"thread_id": "autostream-session-1"}}

#     print("\n" + "=" * 55)
#     print("  AutoStream AI Sales Agent")
#     print("  Powered by LangGraph + Gemini 1.5 Flash")
#     print("=" * 55)
#     print("  Type 'quit' to exit.\n")

#     opener = ("Hi! Welcome to AutoStream — AI-powered video editing "
#               "for content creators. How can I help you today?")
#     print(f"Agent: {opener}\n")

#     # Track if this is the first user turn
#     first_turn = True

#     while True:
#         user_input = input("You: ").strip()
#         if not user_input:
#             continue
#         if user_input.lower() in ("quit", "exit"):
#             print("\nAgent: Thanks for chatting! Have a great day!\n")
#             break

#         if first_turn:
#             # On first turn: pass full initial state + opener + user message together
#             invoke_input = {
#                 "messages"       : [AIMessage(content=opener), HumanMessage(content=user_input)],
#                 "intent"         : "",
#                 "lead_name"      : None,
#                 "lead_email"     : None,
#                 "lead_platform"  : None,
#                 "collecting_lead": False,
#                 "lead_captured"  : False,
#             }
#             first_turn = False
#         else:
#             # Subsequent turns: LangGraph MemorySaver handles history automatically
#             invoke_input = {"messages": [HumanMessage(content=user_input)]}

#         result = graph.invoke(invoke_input, config=config)

#         for msg in reversed(result["messages"]):
#             if isinstance(msg, AIMessage) and msg.content:
#                 print(f"\nAgent: {msg.content}\n")
#                 break


# if __name__ == "__main__":
#     run_agent()