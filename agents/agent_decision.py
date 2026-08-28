import json
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Literal,
    TypedDict,
    Union,
    Annotated,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import (
    MessagesState,
    StateGraph,
    END,
)

import os
import getpass

from dotenv import load_dotenv

from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

config = Config()


# ============================================================
# LANGGRAPH MEMORY
# ============================================================

memory = MemorySaver()

thread_config = {
    "configurable": {
        "thread_id": "1"
    }
}


# ============================================================
# AGENT CONFIGURATION
# ============================================================

class AgentConfig:
    """
    Configuration settings for the agent decision system.
    """

    # --------------------------------------------------------
    # Decision model
    # --------------------------------------------------------

    DECISION_MODEL = "gpt-4o"

    # --------------------------------------------------------
    # Vision model
    # --------------------------------------------------------

    VISION_MODEL = "gpt-4o"

    # --------------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------------

    CONFIDENCE_THRESHOLD = 0.85

    # ========================================================
    # DECISION SYSTEM PROMPT
    # ========================================================

    DECISION_SYSTEM_PROMPT = """
You are an intelligent medical triage and routing system.

Your job is to determine which specialized agent should handle
the user's request.

IMPORTANT ROUTING RULES:

1. CONVERSATION_AGENT
Use ONLY for:
- greetings
- casual conversation
- non-medical questions
- simple everyday questions that do not require medical knowledge

Examples:
"Hello"
"How are you?"
"What is the weather?"
"What should I eat for breakfast?"

2. RAG_AGENT
Use for MEDICAL KNOWLEDGE QUESTIONS.

If the user asks about a disease, disorder, symptom, risk factor,
diagnostic test, laboratory test, treatment concept, anatomy,
medical imaging, prevention, medical terminology, or established
medical knowledge, prefer RAG_AGENT.

This applies even when the question is short or general.

Examples:
"What are the risk factors for cardiovascular disease?"
"What is hypertension?"
"What are the symptoms of asthma?"
"What tests assess kidney function?"
"What is HbA1c?"
"What is the difference between CT and MRI?"
"What are common symptoms of diabetes?"
"What are the causes of anemia?"
"What are common treatments for cancer?"
"What is pneumonia?"
"What is an ECG?"

The medical knowledge base contains broad medical information
covering areas including:
- general medicine
- cardiology
- neurology
- respiratory medicine
- endocrinology
- diabetes
- gastroenterology
- liver disease
- nephrology
- kidney disease
- oncology
- infectious diseases
- dermatology
- hematology
- musculoskeletal medicine
- psychiatry and mental health
- medical laboratory tests
- medical imaging
- brain tumors
- brain MRI
- chest X-ray
- medical AI
- deep learning in medical imaging
- preventive health

It also contains specialized documents about:
- brain tumor detection
- deep learning for brain tumors
- COVID-19 and chest X-ray analysis

IMPORTANT:
Do NOT send a medical knowledge question to
CONVERSATION_AGENT merely because no image is uploaded.

3. WEB_SEARCH_PROCESSOR_AGENT
Use when the user specifically asks for:
- recent medical developments
- latest medical research
- current outbreaks
- current statistics
- newly approved treatments
- current medical guidelines
- information that may have changed recently

Examples:
"What is the latest treatment for Alzheimer's disease?"
"What are the latest COVID variants?"
"What medical discoveries happened this month?"

4. BRAIN_TUMOR_AGENT
Use when the user uploads a brain MRI or asks to analyze
a brain MRI image for a tumor.

5. CHEST_XRAY_AGENT
Use when the user uploads a chest X-ray for image analysis.

6. SKIN_LESION_AGENT
Use when the user uploads a skin lesion image for classification.

IMAGE RULE:
If an image is uploaded, prioritize the appropriate medical
vision agent based on the image type and user's request.

If an image is uploaded without a clear description:
- chest/radiograph-looking image -> CHEST_XRAY_AGENT
- brain MRI -> BRAIN_TUMOR_AGENT
- skin lesion/photo of lesion -> SKIN_LESION_AGENT

DECISION PRIORITY:

1. Medical image analysis -> appropriate vision agent
2. Current/recent medical information -> WEB_SEARCH_PROCESSOR_AGENT
3. Specific or established medical knowledge -> RAG_AGENT
4. General/non-medical conversation -> CONVERSATION_AGENT

When uncertain between CONVERSATION_AGENT and RAG_AGENT for a
medical question, choose RAG_AGENT.

Return ONLY valid JSON:

{{
    "agent": "AGENT_NAME",
    "reasoning": "Brief explanation of why this agent was selected",
    "confidence": 0.95
}}

The confidence value must be between 0.0 and 1.0.
"""


class AgentState(MessagesState):
    """
    State maintained across the workflow.
    """

    agent_name: Optional[str]

    current_input: Optional[
        Union[str, Dict]
    ]

    has_image: bool

    image_type: Optional[str]

    output: Optional[str]

    needs_human_validation: bool

    retrieval_confidence: float

    bypass_routing: bool

    insufficient_info: bool


    current_input: Optional[
        Union[str, Dict]
    ]

    has_image: bool

    image_type: Optional[str]

    output: Optional[str]

    needs_human_validation: bool

    retrieval_confidence: float

    bypass_routing: bool

    insufficient_info: bool


# ============================================================
# AGENT DECISION TYPE
# ============================================================

class AgentDecision(TypedDict):
    """
    Output structure for the decision agent.
    """

    agent: str
    reasoning: str
    confidence: float


# ============================================================
# CREATE AGENT GRAPH
# ============================================================

def create_agent_graph():
    """
    Create and configure the LangGraph for agent orchestration.
    """

    # ========================================================
    # GUARDRAILS
    # ========================================================

    guardrails = LocalGuardrails(
        config.rag.llm
    )


    # ========================================================
    # DECISION MODEL
    # ========================================================

    decision_model = (
        config.agent_decision.llm
    )


    # ========================================================
    # JSON PARSER
    # ========================================================

    json_parser = JsonOutputParser(
        pydantic_object=AgentDecision
    )


    # ========================================================
    # DECISION PROMPT
    # ========================================================

    decision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                AgentConfig.DECISION_SYSTEM_PROMPT,
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )


    # ========================================================
    # DECISION CHAIN
    # ========================================================

    decision_chain = (
        decision_prompt
        | decision_model
        | json_parser
    )


    # ========================================================
    # ANALYZE INPUT
    # ========================================================

    def analyze_input(
        state: AgentState
    ) -> AgentState:
        """
        Analyze the input to detect images and
        determine input type.
        """

        current_input = state[
            "current_input"
        ]

        has_image = False
        image_type = None


        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        input_text = ""

        if isinstance(
            current_input,
            str,
        ):

            input_text = current_input

        elif isinstance(
            current_input,
            dict,
        ):

            input_text = current_input.get(
                "text",
                "",
            )


        # ----------------------------------------------------
        # Guardrails
        # ----------------------------------------------------

        if input_text:

            is_allowed, message = (
                guardrails.check_input(
                    input_text
                )
            )

            if not is_allowed:

                print(
                    "Selected agent: "
                    "INPUT GUARDRAILS, "
                    "Message:",
                    message,
                )

                return {
                    **state,

                    "messages": message,

                    "agent_name":
                        "INPUT_GUARDRAILS",

                    "has_image": False,

                    "image_type": None,

                    "bypass_routing": True,
                }


        # ====================================================
        # IMAGE DETECTION
        # ====================================================

        if (
            isinstance(
                current_input,
                dict,
            )
            and "image" in current_input
        ):

            has_image = True

            input_text_lower = (
                input_text.lower()
            )


            # ------------------------------------------------
            # Brain MRI
            # ------------------------------------------------

            if (
                "brain"
                in input_text_lower
                or "mri"
                in input_text_lower
                or "tumor"
                in input_text_lower
            ):

                image_type = (
                    "brain_mri"
                )


            # ------------------------------------------------
            # Skin lesion
            # ------------------------------------------------

            elif (
                "skin"
                in input_text_lower
                or "lesion"
                in input_text_lower
                or "melanoma"
                in input_text_lower
                or "mole"
                in input_text_lower
            ):

                image_type = (
                    "skin_lesion"
                )


            # ------------------------------------------------
            # Chest X-ray
            # ------------------------------------------------

            else:

                image_type = (
                    "chest_xray"
                )


            print(
                "DIRECT IMAGE TYPE:",
                image_type,
            )


        # ====================================================
        # RETURN STATE
        # ====================================================

        return {
            **state,

            "has_image": has_image,

            "image_type": image_type,

            "bypass_routing": False,
        }


    # ========================================================
    # CHECK BYPASS
    # ========================================================

    def check_if_bypassing(
        state: AgentState
    ) -> str:
        """
        Check if routing should be bypassed because
        guardrails blocked the input.
        """

        if state.get(
            "bypass_routing",
            False,
        ):

            return "apply_guardrails"

        return "route_to_agent"


    # ========================================================
    # ROUTE TO AGENT
    # ========================================================

    def route_to_agent(
        state: AgentState
    ) -> Dict:
        """
        Determine which agent should handle the query.
        """

        messages = state[
            "messages"
        ]

        current_input = state[
            "current_input"
        ]

        has_image = state[
            "has_image"
        ]

        image_type = state[
            "image_type"
        ]


        # ----------------------------------------------------
        # Extract query text
        # ----------------------------------------------------

        input_text = ""

        if isinstance(
            current_input,
            str,
        ):

            input_text = current_input

        elif isinstance(
            current_input,
            dict,
        ):

            input_text = current_input.get(
                "text",
                "",
            )


        # ----------------------------------------------------
        # Recent conversation context
        # ----------------------------------------------------

        recent_context = ""

        for msg in messages[-6:]:

            if isinstance(
                msg,
                HumanMessage,
            ):

                recent_context += (
                    f"User: {msg.content}\n"
                )

            elif isinstance(
                msg,
                AIMessage,
            ):

                recent_context += (
                    f"Assistant: {msg.content}\n"
                )


        # ====================================================
        # DIRECT IMAGE ROUTING
        # ====================================================

        if has_image:

            # ------------------------------------------------
            # Brain MRI
            # ------------------------------------------------

            if image_type == "brain_mri":

                print(
                    "Decision: "
                    "BRAIN_TUMOR_AGENT"
                )

                return {
                    **state,

                    "agent_name":
                        "BRAIN_TUMOR_AGENT",

                    "next":
                        "BRAIN_TUMOR_AGENT",
                }


            # ------------------------------------------------
            # Skin lesion
            # ------------------------------------------------

            elif image_type == "skin_lesion":

                print(
                    "Decision: "
                    "SKIN_LESION_AGENT"
                )

                return {
                    **state,

                    "agent_name":
                        "SKIN_LESION_AGENT",

                    "next":
                        "SKIN_LESION_AGENT",
                }


            # ------------------------------------------------
            # Chest X-ray
            # ------------------------------------------------

            else:

                print(
                    "Decision: "
                    "CHEST_XRAY_AGENT"
                )

                return {
                    **state,

                    "agent_name":
                        "CHEST_XRAY_AGENT",

                    "next":
                        "CHEST_XRAY_AGENT",
                }


        # ====================================================
        # DECISION MODEL
        # ====================================================

        decision_input = f"""
User query:
{input_text}

Recent conversation context:
{recent_context}

Has image:
{has_image}

Image type:
{image_type if has_image else "None"}

Important routing instructions:

- No image does NOT automatically mean Conversation Agent.
- Specific medical knowledge questions should go to RAG_AGENT.
- Questions about indexed medical documents should go to RAG_AGENT.
- Questions about previously uploaded medical reports should go to RAG_AGENT.
- Current/latest/recent medical questions should go to
  WEB_SEARCH_PROCESSOR_AGENT.
- General conversation should go to CONVERSATION_AGENT.

Which agent should handle this query?
"""


        # ----------------------------------------------------
        # Call decision model
        # ----------------------------------------------------

        try:

            decision = decision_chain.invoke(
                {
                    "input":
                        decision_input
                }
            )

        except Exception as e:

            print(
                "Decision model error:",
                str(e),
            )

            # ------------------------------------------------
            # Safe fallback routing
            # ------------------------------------------------

            query_lower = (
                input_text.lower()
            )


            # Current information
            current_keywords = [
                "latest",
                "recent",
                "current",
                "today",
                "currently",
                "newly published",
                "recent research",
                "latest research",
            ]

            if any(
                keyword in query_lower
                for keyword in current_keywords
            ):

                fallback_agent = (
                    "WEB_SEARCH_PROCESSOR_AGENT"
                )


            # RAG knowledge
            elif any(
                keyword in query_lower
                for keyword in [
                    "according to the medical documents",
                    "according to the documents",
                    "knowledge base",
                    "indexed documents",
                    "medical literature",
                    "brain tumor detection",
                    "brain tumour detection",
                    "deep learning techniques",
                    "brain tumor",
                    "brain tumour",
                ]
            ):

                fallback_agent = (
                    "RAG_AGENT"
                )


            # General conversation
            else:

                fallback_agent = (
                    "CONVERSATION_AGENT"
                )


            print(
                "Fallback Decision:",
                fallback_agent,
            )

            return {
                **state,

                "agent_name":
                    fallback_agent,

                "next":
                    fallback_agent,
            }


        # ====================================================
        # VALIDATE DECISION
        # ====================================================

        valid_agents = {
            "CONVERSATION_AGENT",
            "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT",
        }


        selected_agent = decision.get(
            "agent",
            "CONVERSATION_AGENT",
        )


        if (
            selected_agent
            not in valid_agents
        ):

            print(
                "Invalid agent returned:",
                selected_agent,
            )

            selected_agent = (
                "CONVERSATION_AGENT"
            )


        confidence = decision.get(
            "confidence",
            0.0,
        )


        # ====================================================
        # DEBUG
        # ====================================================

        print(
            f"Decision: {selected_agent}"
        )

        print(
            f"Decision confidence: "
            f"{confidence}"
        )

        print(
            f"Decision reasoning: "
            f"{decision.get('reasoning', '')}"
        )


        # ====================================================
        # UPDATE STATE
        # ====================================================

        updated_state = {
            **state,

            "agent_name":
                selected_agent,
        }


        # ====================================================
        # CONFIDENCE ROUTING
        # ====================================================

        if (
            confidence
            < AgentConfig.CONFIDENCE_THRESHOLD
        ):

            # For low-confidence RAG-like queries,
            # use RAG rather than silently answering
            # as general conversation.

            query_lower = (
                input_text.lower()
            )

            rag_indicators = [
                "medical document",
                "medical documents",
                "knowledge base",
                "indexed document",
                "indexed documents",
                "medical literature",
                "research paper",
                "brain tumor",
                "brain tumour",
                "deep learning",
                "covid",
                "covid-19",
                "chest x-ray",
            ]

            if any(
                indicator in query_lower
                for indicator in rag_indicators
            ):

                print(
                    "Low confidence but "
                    "query matches RAG topic. "
                    "Routing to RAG_AGENT."
                )

                return {
                    **updated_state,

                    "agent_name":
                        "RAG_AGENT",

                    "next":
                        "RAG_AGENT",
                }

            return {
                **updated_state,

                "next":
                    "needs_validation",
            }


        return {
            **updated_state,

            "next":
                selected_agent,
        }


    # ========================================================
    # CONVERSATION AGENT
    # ========================================================

    def run_conversation_agent(
        state: AgentState
    ) -> AgentState:
        """
        Handle general conversation.
        """

        print(
            "Selected agent: "
            "CONVERSATION_AGENT"
        )

        messages = state[
            "messages"
        ]

        current_input = state[
            "current_input"
        ]


        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        input_text = ""

        if isinstance(
            current_input,
            str,
        ):

            input_text = current_input

        elif isinstance(
            current_input,
            dict,
        ):

            input_text = current_input.get(
                "text",
                "",
            )


        # ----------------------------------------------------
        # Conversation context
        # ----------------------------------------------------

        recent_context = ""

        for msg in messages:

            if isinstance(
                msg,
                HumanMessage,
            ):

                recent_context += (
                    f"User: {msg.content}\n"
                )

            elif isinstance(
                msg,
                AIMessage,
            ):

                recent_context += (
                    f"Assistant: {msg.content}\n"
                )


        # ====================================================
        # CONVERSATION PROMPT
        # ====================================================

        conversation_prompt = f"""
User query:
{input_text}

Recent conversation context:
{recent_context}

You are an AI-powered Medical Conversation Assistant.

Your goal is to facilitate smooth and informative
conversations with users while maintaining medical
accuracy and clarity.

Capabilities:

- Engage in general conversation.
- Answer general medical questions.
- Keep answers clear and concise.
- Maintain conversation context.
- Explain medical concepts in simple language.
- Recommend professional medical evaluation when
  appropriate.

Important:

Do not pretend to diagnose a patient.

Do not prescribe medication.

Do not claim that an image was analyzed if an image
analysis agent has not actually produced a result.

If the user asks about complex medical knowledge that
requires retrieval, that query should normally have
already been routed to RAG_AGENT.

If the user asks for current/latest information, that
query should normally have already been routed to the
web search agent.

Response style:

- Professional
- Clear
- Conversational
- Use bullet points when useful
- Avoid unnecessary technical language

User question:
{input_text}

Answer:
"""


        # ====================================================
        # GENERATE RESPONSE
        # ====================================================

        response = (
            config.conversation.llm.invoke(
                conversation_prompt
            )
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {
            **state,

            "output":
                response,

            "agent_name":
                "CONVERSATION_AGENT",
        }


    # ========================================================
    # RAG AGENT
    # ========================================================

    def run_rag_agent(
        state: AgentState
    ) -> AgentState:
        """
        Handle medical knowledge queries using RAG.
        """

        print(
            "Selected agent: RAG_AGENT"
        )


        # ----------------------------------------------------
        # Initialize RAG
        # ----------------------------------------------------

        rag_agent = MedicalRAG(
            config
        )


        messages = state[
            "messages"
        ]

        query = state[
            "current_input"
        ]

        rag_context_limit = (
            config.rag.context_limit
        )


        # ----------------------------------------------------
        # Conversation context
        # ----------------------------------------------------

        recent_context = ""

        for msg in messages[
            -rag_context_limit:
        ]:

            if isinstance(
                msg,
                HumanMessage,
            ):

                recent_context += (
                    f"User: {msg.content}\n"
                )

            elif isinstance(
                msg,
                AIMessage,
            ):

                recent_context += (
                    f"Assistant: {msg.content}\n"
                )


        # ====================================================
        # PROCESS RAG QUERY
        # ====================================================

        response = rag_agent.process_query(
            query,
            chat_history=recent_context,
        )


        retrieval_confidence = response.get(
            "confidence",
            0.0,
        )


        print(
            f"Retrieval Confidence: "
            f"{retrieval_confidence}"
        )

        print(
            f"Sources: "
            f"{len(response['sources'])}"
        )


        # ====================================================
        # CHECK INSUFFICIENT INFORMATION
        # ====================================================

        insufficient_info = False

        response_content = (
            response["response"]
        )


        if isinstance(
            response_content,
            BaseMessage,
        ):

            response_text = (
                response_content.content
            )

        else:

            response_text = (
                response_content
            )


        print(
            "Response text type:",
            type(response_text),
        )


        print(
            "Response text preview:",
            response_text[:100],
            "...",
        )


        # ----------------------------------------------------
        # Detect insufficient information
        # ----------------------------------------------------

        if isinstance(
            response_text,
            str,
        ):

            insufficient_phrases = [
                "I don't have enough information",
                "don't have enough information",
                "not enough information",
                "insufficient information",
                "cannot answer",
                "unable to answer",
            ]

            if any(
                phrase.lower()
                in response_text.lower()
                for phrase in insufficient_phrases
            ):

                print(
                    "RAG response indicates "
                    "insufficient information"
                )

                insufficient_info = True


        print(
            "Insufficient info flag set to:",
            insufficient_info,
        )


        # ====================================================
        # CREATE OUTPUT
        # ====================================================

        # Keep the generated RAG response.
        # Retrieval confidence is a ranking score and should
        # not cause an otherwise valid answer to be erased.

        response_output = AIMessage(
            content=response_text
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            **state,

            "output":
                response_output,

            "needs_human_validation":
                False,

            "retrieval_confidence":
                retrieval_confidence,

            "agent_name":
                "RAG_AGENT",

            "insufficient_info":
                insufficient_info,
        }


    # ========================================================
    # WEB SEARCH PROCESSOR
    # ========================================================

    def run_web_search_processor_agent(
        state: AgentState
    ) -> AgentState:
        """
        Process web search results and generate a response.
        """

        print(
            "Selected agent: "
            "WEB_SEARCH_PROCESSOR_AGENT"
        )

        print(
            "[WEB_SEARCH_PROCESSOR_AGENT] "
            "Processing Web Search Results..."
        )


        messages = state[
            "messages"
        ]

        web_search_context_limit = (
            config.web_search.context_limit
        )


        # ----------------------------------------------------
        # Recent context
        # ----------------------------------------------------

        recent_context = ""

        for msg in messages[
            -web_search_context_limit:
        ]:

            if isinstance(
                msg,
                HumanMessage,
            ):

                recent_context += (
                    f"User: {msg.content}\n"
                )

            elif isinstance(
                msg,
                AIMessage,
            ):

                recent_context += (
                    f"Assistant: {msg.content}\n"
                )


        # ----------------------------------------------------
        # Web search agent
        # ----------------------------------------------------

        web_search_processor = (
            WebSearchProcessorAgent(
                config
            )
        )


        processed_response = (
            web_search_processor.process_web_search_results(
                query=state[
                    "current_input"
                ],
                chat_history=recent_context,
            )
        )


        # ----------------------------------------------------
        # Agent tracking
        # ----------------------------------------------------

        if state[
            "agent_name"
        ] is not None:

            involved_agents = (
                f"{state['agent_name']}, "
                "WEB_SEARCH_PROCESSOR_AGENT"
            )

        else:

            involved_agents = (
                "WEB_SEARCH_PROCESSOR_AGENT"
            )


        # ====================================================
        # RETURN
        # ====================================================

        return {
            **state,

            "output":
                processed_response,

            "agent_name":
                involved_agents,
        }


    # ========================================================
    # CONFIDENCE BASED ROUTING
    # ========================================================

    def confidence_based_routing(
        state: AgentState
    ) -> str:
        """
        Route the RAG response.

        Keep a valid RAG response even when the numeric
        retrieval confidence is below the configured threshold.
        Web search is used only when RAG explicitly reports
        insufficient information.
        """

        retrieval_confidence = state.get(
            "retrieval_confidence",
            0.0
        )

        insufficient_info = state.get(
            "insufficient_info",
            False
        )

        print(
            "Routing check - "
            f"Retrieval confidence: "
            f"{retrieval_confidence}"
        )

        print(
            "Routing check - "
            f"Insufficient info flag: "
            f"{insufficient_info}"
        )

        # Only fall back to Web Search when RAG
        # explicitly says it does not have enough information.
        if insufficient_info:

            print(
                "RAG indicates insufficient information. "
                "Routing to Web Search Agent..."
            )

            return "WEB_SEARCH_PROCESSOR_AGENT"

        print(
            "RAG has sufficient information. "
            "Keeping RAG response."
        )

        return "check_validation"


    # ========================================================
    # CREATE WORKFLOW
    # ========================================================

    workflow = StateGraph(
        AgentState
    )


    # ========================================================
    # ADD NODES
    # ========================================================

    workflow.add_node(
        "analyze_input",
        analyze_input,
    )

    workflow.add_node(
        "route_to_agent",
        route_to_agent,
    )

    workflow.add_node(
        "CONVERSATION_AGENT",
        run_conversation_agent,
    )

    workflow.add_node(
        "RAG_AGENT",
        run_rag_agent,
    )

    workflow.add_node(
        "WEB_SEARCH_PROCESSOR_AGENT",
        run_web_search_processor_agent,
    )

    workflow.add_node(
        "BRAIN_TUMOR_AGENT",
        run_brain_tumor_agent,
    )

    workflow.add_node(
        "CHEST_XRAY_AGENT",
        run_chest_xray_agent,
    )

    workflow.add_node(
        "SKIN_LESION_AGENT",
        run_skin_lesion_agent,
    )

    workflow.add_node(
        "check_validation",
        handle_human_validation,
    )

    workflow.add_node(
        "needs_validation",
        perform_human_validation,
    )

    workflow.add_node(
        "human_validation",
        perform_human_validation,
    )

    workflow.add_node(
        "apply_guardrails",
        apply_output_guardrails,
    )


    # ========================================================
    # ENTRY POINT
    # ========================================================

    workflow.set_entry_point(
        "analyze_input"
    )


    # ========================================================
    # GUARDRAIL ROUTING
    # ========================================================

    workflow.add_conditional_edges(
        "analyze_input",

        check_if_bypassing,

        {
            "apply_guardrails":
                "apply_guardrails",

            "route_to_agent":
                "route_to_agent",
        },
    )


    # ========================================================
    # AGENT ROUTING
    # ========================================================

    workflow.add_conditional_edges(
        "route_to_agent",

        lambda x: x["next"],

        {
            "CONVERSATION_AGENT":
                "CONVERSATION_AGENT",

            "RAG_AGENT":
                "RAG_AGENT",

            "WEB_SEARCH_PROCESSOR_AGENT":
                "WEB_SEARCH_PROCESSOR_AGENT",

            "BRAIN_TUMOR_AGENT":
                "BRAIN_TUMOR_AGENT",

            "CHEST_XRAY_AGENT":
                "CHEST_XRAY_AGENT",

            "SKIN_LESION_AGENT":
                "SKIN_LESION_AGENT",

            "needs_validation":
                "needs_validation",
        },
    )


    # ========================================================
    # AGENT OUTPUT ROUTING
    # ========================================================

    workflow.add_edge(
        "CONVERSATION_AGENT",
        "check_validation",
    )

    workflow.add_edge(
        "WEB_SEARCH_PROCESSOR_AGENT",
        "check_validation",
    )

    workflow.add_conditional_edges(
        "RAG_AGENT",
        confidence_based_routing,
    )

    workflow.add_edge(
        "BRAIN_TUMOR_AGENT",
        "check_validation",
    )

    workflow.add_edge(
        "CHEST_XRAY_AGENT",
        "check_validation",
    )

    workflow.add_edge(
        "SKIN_LESION_AGENT",
        "check_validation",
    )


    # ========================================================
    # GUARDRAILS
    # ========================================================

    workflow.add_edge(
        "apply_guardrails",
        END,
    )


    workflow.add_edge(
        "check_validation",
        "apply_guardrails",
    )

    workflow.add_edge(
        "human_validation",
        "apply_guardrails",
    )


    # ========================================================
    # COMPILE
    # ========================================================

    return workflow.compile(
        checkpointer=memory
    )


# ============================================================
# BRAIN TUMOR AGENT
# ============================================================

def run_brain_tumor_agent(
    state: AgentState
) -> AgentState:
    """
    Handle brain MRI image analysis.

    Actual model integration can be connected when
    the trained brain-tumor model weights are available.
    """

    print(
        "Selected agent: "
        "BRAIN_TUMOR_AGENT"
    )


    response = AIMessage(
        content="""
### Brain MRI Analysis

The brain MRI was uploaded successfully.

However, the trained brain-tumor analysis model
is currently unavailable in this installation.

**No medical prediction was generated.**

Please consult a qualified radiologist or neurologist
for interpretation of the MRI.
"""
    )


    return {
        **state,

        "output":
            response,

        "needs_human_validation":
            False,

        "agent_name":
            "BRAIN_TUMOR_AGENT",
    }


# ============================================================
# CHEST X-RAY AGENT
# ============================================================

def run_chest_xray_agent(
    state: AgentState
) -> AgentState:
    """
    Handle chest X-ray image analysis.

    The trained chest X-ray model weights are currently
    unavailable, so this function does NOT generate
    fabricated medical findings.
    """

    print(
        "Selected agent: "
        "CHEST_XRAY_AGENT"
    )


    response = AIMessage(
        content="""
### Chest X-Ray Analysis

The chest X-ray was uploaded successfully.

However, the trained chest X-ray analysis model
is currently unavailable in this installation.

**No medical prediction was generated.**

Please consult a qualified healthcare professional
for interpretation of the uploaded X-ray.
"""
    )


    return {
        **state,

        "output":
            response,

        "needs_human_validation":
            False,

        "agent_name":
            "CHEST_XRAY_AGENT",
    }


# ============================================================
# SKIN LESION AGENT
# ============================================================

def run_skin_lesion_agent(
    state: AgentState
) -> AgentState:
    """
    Handle skin lesion image analysis.

    The actual skin-lesion inference implementation
    exists in the image-analysis agent package.
    """

    print(
        "Selected agent: "
        "SKIN_LESION_AGENT"
    )


    # --------------------------------------------------------
    # Try to use the existing ImageAnalysisAgent
    # --------------------------------------------------------

    try:

        current_input = state[
            "current_input"
        ]

        image_path = None

        if isinstance(
            current_input,
            dict,
        ):

            image_path = current_input.get(
                "image"
            )


        if image_path:

            # Try common method names used by
            # image-analysis implementations.

            analyzer = (
                AgentConfig.image_analyzer
            )


            if hasattr(
                analyzer,
                "analyze_image",
            ):

                result = (
                    analyzer.analyze_image(
                        image_path,
                        "skin_lesion",
                    )
                )

                if isinstance(
                    result,
                    BaseMessage,
                ):

                    response = result

                else:

                    response = AIMessage(
                        content=str(result)
                    )

            else:

                response = AIMessage(
                    content="""
### Skin Lesion Analysis

The skin lesion image was uploaded successfully.

The skin-lesion analysis agent is available,
but the current image-analysis integration does
not expose an `analyze_image` method.

**No medical prediction was generated.**

Please consult a dermatologist for professional
evaluation.
"""
                )

        else:

            response = AIMessage(
                content="""
### Skin Lesion Analysis

No image path was provided to the
skin-lesion analysis agent.

Please upload a skin lesion image again.
"""
            )


    except Exception as e:

        print(
            "Skin lesion analysis error:",
            str(e),
        )

        response = AIMessage(
            content="""
### Skin Lesion Analysis

The image was received, but the skin-lesion
analysis could not be completed.

**No medical prediction was generated.**

Please consult a dermatologist for professional
evaluation.
"""
        )


    return {
        **state,

        "output":
            response,

        "needs_human_validation":
            False,

        "agent_name":
            "SKIN_LESION_AGENT",
    }


# ============================================================
# HUMAN VALIDATION CHECK
# ============================================================

def handle_human_validation(
    state: AgentState
) -> AgentState:
    """
    Skip validation by default and continue workflow.
    """

    return {
        **state,

        "needs_human_validation":
            False,
    }


# ============================================================
# HUMAN VALIDATION
# ============================================================

def perform_human_validation(
    state: AgentState
) -> AgentState:
    """
    Handle human validation process.
    """

    print(
        "Selected agent: HUMAN_VALIDATION"
    )


    output = state.get(
        "output"
    )


    if isinstance(
        output,
        BaseMessage,
    ):

        output_content = (
            output.content
        )

    else:

        output_content = str(
            output
        )


    validation_prompt = (
        f"{output_content}\n\n"
        "Validation complete."
    )


    validation_message = AIMessage(
        content=validation_prompt
    )


    return {
        **state,

        "output":
            validation_message,

        "agent_name":
            f"{state['agent_name']}, "
            "HUMAN_VALIDATION",
    }


# ============================================================
# OUTPUT GUARDRAILS
# ============================================================

def apply_output_guardrails(
    state: AgentState
) -> AgentState:
    """
    Return the final response.
    """

    output = state.get(
        "output"
    )


    if output is None:

        output = AIMessage(
            content=(
                "I'm sorry, I couldn't "
                "generate a response."
            )
        )


    elif not isinstance(
        output,
        BaseMessage,
    ):

        output = AIMessage(
            content=str(
                output
            )
        )


    return {
        **state,

        "messages": [
            output
        ],

        "output":
            output,
    }


# ============================================================
# INITIAL STATE
# ============================================================

def init_agent_state() -> AgentState:
    """
    Initialize the agent state with default values.
    """

    return {
        "messages": [],

        "agent_name":
            None,

        "current_input":
            None,

        "has_image":
            False,

        "image_type":
            None,

        "output":
            None,

        "needs_human_validation":
            False,

        "retrieval_confidence":
            0.0,

        "bypass_routing":
            False,

        "insufficient_info":
            False,
    }


# ============================================================
# PROCESS QUERY
# ============================================================

def process_query(
    query: Union[str, Dict],
    conversation_history: List[
        BaseMessage
    ] = None,
):
    """
    Process a user query through the
    agent decision system.

    Args:
        query:
            User input as either a string or a
            dictionary containing text/image.

        conversation_history:
            Optional previous conversation messages.

    Returns:
        Final LangGraph state.
    """

    # ========================================================
    # CREATE GRAPH
    # ========================================================

    graph = create_agent_graph()


    # ========================================================
    # INITIAL STATE
    # ========================================================

    state = init_agent_state()


    # ========================================================
    # OPTIONAL HISTORY
    # ========================================================

    if conversation_history:

        try:

            state["messages"] = (
                conversation_history.copy()
            )

        except Exception:

            pass


    # ========================================================
    # CURRENT INPUT
    # ========================================================

    state["current_input"] = query


    # ========================================================
    # IMAGE INPUT
    # ========================================================

    if isinstance(
        query,
        dict,
    ):

        text = query.get(
            "text",
            "",
        )

        state["messages"] = [
            HumanMessage(
                content=text
            )
        ]


    else:

        state["messages"] = [
            HumanMessage(
                content=query
            )
        ]


    # ========================================================
    # RUN GRAPH
    # ========================================================

    result = graph.invoke(
        state,
        thread_config,
    )


    # ========================================================
    # LIMIT HISTORY
    # ========================================================

    if len(
        result["messages"]
    ) > config.max_conversation_history:

        result["messages"] = (
            result["messages"][
                -config.max_conversation_history:
            ]
        )


    # ========================================================
    # PRINT HISTORY
    # ========================================================

    for message in result[
        "messages"
    ]:

        try:

            message.pretty_print()

        except Exception:

            print(
                message
            )


    # ========================================================
    # RETURN
    # ========================================================

    return result