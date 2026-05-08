"""Agentic and deterministic report-analysis pipelines for MedAI."""

import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from jinja2 import Template

from src.model import get_model_and_tokenizer

try:
    from src.rag_pipeline import MedicalRAG
except ImportError:
    MedicalRAG = None

try:
    from langchain.agents import AgentExecutor, create_react_agent, tool
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    AgentExecutor = object
    create_react_agent = None
    ConversationBufferWindowMemory = None
    PromptTemplate = None
    ChatVertexAI = None

    class LocalTool:
        """Minimal callable tool wrapper used when LangChain imports are unavailable."""

        def __init__(self, function: Callable[..., str]) -> None:
            """Store the wrapped function and expose LangChain-like metadata."""
            self.function = function
            self.name = function.__name__
            self.description = function.__doc__ or ""

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            """Call the wrapped function directly."""
            return self.function(*args, **kwargs)

        def invoke(self, tool_input: Any) -> str:
            """Call the wrapped function through a LangChain-like invoke method."""
            if isinstance(tool_input, dict):
                return self.function(**tool_input)
            return self.function(tool_input)

    def tool(function: Callable[..., str]) -> LocalTool:
        """Wrap a function as a minimal local tool when LangChain is not importable."""
        return LocalTool(function)

try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None


DEFAULT_VERTEX_MODEL = "gemini-2.5-flash-lite-preview-06-17"
MODEL_CACHE: Dict[str, object] = {"model": None, "tokenizer": None, "device": None}
RAG_CACHE: Dict[str, object] = {"rag": None}


def _load_prediction_components() -> Dict[str, object]:
    """Load and cache the trained MedAI model and tokenizer for tool calls."""
    if MODEL_CACHE["model"] is not None and MODEL_CACHE["tokenizer"] is not None:
        return MODEL_CACHE

    device = os.getenv("MEDAI_DEVICE", "cpu")
    checkpoint_path = "models/best_model.pt"
    model_path = checkpoint_path if os.path.exists(checkpoint_path) else None
    model, tokenizer = get_model_and_tokenizer(model_path=model_path, device=device)
    model.eval()
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["device"] = device
    return MODEL_CACHE


def _load_rag() -> Optional[MedicalRAG]:
    """Load and cache the medical knowledge FAISS index when it exists."""
    if RAG_CACHE["rag"] is not None:
        return RAG_CACHE["rag"]
    if MedicalRAG is None:
        return None

    index_path = "data/medical_kb/medical.index"
    chunks_path = "data/medical_kb/chunks.pkl"
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None

    rag = MedicalRAG(index_path=index_path, chunks_path=chunks_path)
    RAG_CACHE["rag"] = rag
    return rag


@tool
def run_ner(text: str) -> str:
    """Extract medical entities and risk probabilities from report text."""
    try:
        components = _load_prediction_components()
        model = components["model"]
        tokenizer = components["tokenizer"]
        device = str(components["device"])
        prediction = model.predict(text, tokenizer, device)
        return json.dumps(
            {
                "entities": prediction["entities"],
                "risk_level": prediction["risk_level"],
                "risk_probs": prediction["risk_probs"],
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": f"Medical model unavailable: {exc}"})


@tool
def get_risk_level(text: str) -> str:
    """Return the predicted risk level and class probabilities for report text."""
    try:
        components = _load_prediction_components()
        model = components["model"]
        tokenizer = components["tokenizer"]
        device = str(components["device"])
        prediction = model.predict(text, tokenizer, device)
        risk_level = str(prediction["risk_level"])
        probabilities = prediction["risk_probs"]
        confidence = float(probabilities.get(risk_level, 0.0))
        formatted_probabilities = ", ".join(
            f"{label}: {float(probability) * 100:.0f}%"
            for label, probability in probabilities.items()
        )
        return (
            f"Risk Level: {risk_level} (Confidence: {confidence * 100:.0f}%). "
            f"Probabilities: {formatted_probabilities}"
        )
    except Exception as exc:
        return f"Risk assessment unavailable: {exc}"


@tool
def retrieve_medical(query: str) -> str:
    """Retrieve medical knowledge chunks that are semantically relevant to the query."""
    try:
        rag = _load_rag()
        if rag is None:
            return "Medical knowledge base not available. Please run /build-index first."
        context = rag.get_context(query)
        if not context:
            return "No highly relevant medical knowledge was found for this query."
        return context
    except Exception as exc:
        return f"Medical retrieval unavailable: {exc}"


@tool
def format_report(entities: str, risk: str, context: str) -> str:
    """Format entity, risk, and context tool outputs into a patient-friendly report."""
    return (
        "## Key Findings\n\n"
        f"{entities}\n\n"
        "## Risk Assessment\n\n"
        f"{risk}\n\n"
        "This risk assessment is an AI estimate and not a doctor's diagnosis.\n\n"
        "## Medical Context\n\n"
        f"{context}\n\n"
        "## Disclaimer\n\n"
        "This is educational information only and is not a substitute for professional medical advice, "
        "diagnosis, or treatment. Please consult a qualified clinician about your results."
    )


def get_agent() -> AgentExecutor:
    """Create and return the LangChain ReAct agent configured for Vertex AI Gemini."""
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_name = os.getenv("VERTEX_AI_MODEL_NAME", DEFAULT_VERTEX_MODEL)

    if not project_id:
        print("Warning: GOOGLE_CLOUD_PROJECT_ID is not set. Agent will use fallback pipeline.")
        raise RuntimeError("GOOGLE_CLOUD_PROJECT_ID is required for Vertex AI agent execution")

    if (
        create_react_agent is None
        or ConversationBufferWindowMemory is None
        or PromptTemplate is None
        or ChatVertexAI is None
    ):
        print("Warning: LangChain Vertex AI agent dependencies are unavailable. Agent will use fallback pipeline.")
        raise RuntimeError("LangChain Vertex AI agent dependencies are required for agent execution")

    if aiplatform is None:
        print("Warning: google-cloud-aiplatform is not installed. Agent will use fallback pipeline.")
        raise RuntimeError("google-cloud-aiplatform is required for Vertex AI agent execution")

    try:
        aiplatform.init(project=project_id, location=location)
    except Exception as exc:
        print(f"Warning: Vertex AI initialization failed. Agent will use fallback pipeline: {exc}")
        raise

    llm = ChatVertexAI(
        model_name=model_name,
        project=project_id,
        location=location,
        temperature=0.2,
        max_output_tokens=2048,
    )
    tools = [run_ner, get_risk_level, retrieve_medical, format_report]
    system_prompt = """You are MedAI, a medical report analysis assistant. You help patients understand their lab reports in plain English.

You have access to 4 tools:
1. run_ner - extracts medical entities from reports
2. get_risk_level - assesses clinical risk level
3. retrieve_medical - searches medical knowledge base
4. format_report - formats final patient-friendly report

ALWAYS follow this order:
1. First, extract entities using run_ner
2. Then, assess risk using get_risk_level
3. Then, retrieve medical context using retrieve_medical
4. Finally, format the report using format_report

Rules:
- Always ground explanations in retrieved medical knowledge
- Never make up information
- Clearly state that risk assessment is an AI estimate, not a doctor's diagnosis
- Be empathetic and use simple language
"""
    prompt = PromptTemplate.from_template(
        system_prompt
        + """

Previous conversation:
{chat_history}

You can use these tools:
{tools}

Use this format:
Question: the user request
Thought: what you should do next
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... repeat Thought/Action/Action Input/Observation as needed
Thought: I now know the final answer
Final Answer: the final patient-friendly response

Question: {input}
Thought:{agent_scratchpad}
"""
    )
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=False,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        verbose=False,
    )


def run_agent_query(
    agent: Optional[AgentExecutor],
    query: str,
    chat_history: Optional[List] = None,
) -> Dict:
    """Run a user query through the agent and return response metadata."""
    session_id = str(uuid.uuid4())
    input_payload: Dict[str, object] = {"input": query}
    if isinstance(chat_history, str):
        session_id = chat_history
    elif chat_history is not None:
        input_payload["chat_history"] = chat_history

    if agent is None:
        fallback = fallback_pipeline(query)
        fallback["session_id"] = session_id
        return fallback

    try:
        result = agent.invoke(input_payload)
        tools_used: List[str] = []
        for intermediate_step in result.get("intermediate_steps", []):
            action = intermediate_step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name and tool_name not in tools_used:
                tools_used.append(str(tool_name))
        return {
            "response": str(result.get("output", "")),
            "tools_used": tools_used,
            "session_id": session_id,
        }
    except Exception as exc:
        print(f"Warning: Agent execution failed, using fallback pipeline: {exc}")
        fallback = fallback_pipeline(query)
        fallback["session_id"] = session_id
        return fallback


def fallback_pipeline(text: str) -> Dict:
    """Run deterministic report analysis without calling an LLM."""
    entities = run_ner.invoke(text)
    risk = get_risk_level.invoke(text)
    context = retrieve_medical.invoke(text)
    template = Template(
        """## Your Medical Report Analysis

### Key Findings
{{ entities }}

### Risk Assessment
{{ risk }}

### What This Means
{{ context }}

---
⚠️ This is an AI-generated analysis for educational purposes only. It is NOT a substitute for professional medical advice. Please consult your doctor.
"""
    )
    return {
        "response": template.render(entities=entities, risk=risk, context=context),
        "tools_used": ["run_ner", "get_risk_level", "retrieve_medical", "format_report"],
        "fallback": True,
    }
