import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_llm(temp=0.1):
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=temp
    )

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = create_llm()

class ConversationConfig:
    def __init__(self):
        self.llm = create_llm()

class WebSearchConfig:
    def __init__(self):
        self.llm = create_llm()
        self.context_limit = 20

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        # Dense embedding model used by the Qdrant RAG pipeline.
        # BGE-small produces 384-dimensional vectors and is lightweight enough
        # for a college/laptop setup. The model is downloaded automatically on
        # first use by LangChain/FastEmbed.
        self.embedding_model_name = os.getenv(
            "RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self.embedding_dim = 384
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"

        self.collection_name = "medical_assistance_rag"

        self.chunk_size = 512
        self.chunk_overlap = 50

        self.embedding_model = None  # initialized by VectorStore

        self.llm = create_llm()
        self.summarizer_model = create_llm()
        self.chunker_model = create_llm()
        self.response_generator_model = create_llm()

        self.top_k = 10
        self.vector_search_type = 'similarity'

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 5

        self.max_context_length = 8192

        self.include_sources = True

        self.min_retrieval_confidence = 0.40

        self.context_limit = 20

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"

        self.llm = create_llm()

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": False,
            "CHEST_XRAY_AGENT": False,
            "SKIN_LESION_AGENT": False
        }

        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = False
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()

        self.max_conversation_history = 20

