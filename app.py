import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Request,
    Response,
    Cookie,
)
from groq import RateLimitError
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
from agents.agent_decision import process_query
from agents.rag_agent.doc_parser import MedicalDocParser


# ============================================================
# MEDICAL REPORT SESSION STORAGE
# ============================================================

# Stores the extracted text of the latest uploaded medical
# report for each browser session.
#
# This allows follow-up questions about the same report
# without uploading the PDF again.
REPORT_SESSIONS: Dict[str, str] = {}


# ============================================================
# CONFIGURATION
# ============================================================

config = Config()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MEDEASY - Multi-Agent Medical Assistant",
    version="2.0",
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"
MEDICAL_REPORT_FOLDER = "uploads/medical_reports"

for directory in [
    UPLOAD_FOLDER,
    FRONTEND_UPLOAD_FOLDER,
    SKIN_LESION_OUTPUT,
    SPEECH_DIR,
    MEDICAL_REPORT_FOLDER,
]:
    os.makedirs(directory, exist_ok=True)



# ============================================================
# INITIALIZE RAG DATABASE IF NOT PRESENT
# ============================================================

QDRANT_DB_PATH = "data/qdrant_db"

if not os.path.exists(QDRANT_DB_PATH):
    print("RAG database not found. Building knowledge base...")

    import subprocess

    subprocess.run(
        [
            "python",
            "ingest_rag_data.py",
            "--dir",
            "data/medical_documents",
        ],
        check=True,
    )

    print("RAG knowledge base created successfully.")
else:
    print("RAG database already exists. Skipping ingestion.")


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/data",
    StaticFiles(directory="data"),
    name="data",
)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# ELEVENLABS
# ============================================================

client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "pdf",
}


def allowed_file(filename: str) -> bool:
    """
    Check whether a file has an allowed extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# AUDIO CLEANUP
# ============================================================

def cleanup_old_audio():
    """
    Delete old MP3 files.
    """

    try:
        files = glob.glob(
            f"{SPEECH_DIR}/*.mp3"
        )

        for file in files:
            os.remove(file)

    except Exception as e:
        print(
            f"Error during cleanup: {e}"
        )


cleanup_old_audio()


# ============================================================
# REQUEST MODELS
# ============================================================

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []


class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def index(request: Request):
    """
    Serve the MEDEASY frontend.
    """

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy"
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(
    request: QueryRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None),
):
    """
    Process a normal text query through the multi-agent system.

    If the current browser session has an uploaded medical
    report, its extracted text is included as context.
    """

    # Create a session ID if this is the first request.
    if not session_id:
        session_id = str(uuid.uuid4())

    try:

        # ----------------------------------------------------
        # Get stored report for this browser session
        # ----------------------------------------------------

        report_context = REPORT_SESSIONS.get(
            session_id
        )

        query = request.query

        # ----------------------------------------------------
        # Add report context to follow-up questions
        # ----------------------------------------------------

        if report_context:

            query = f"""
The user previously uploaded a medical report.

Use the following medical report as context when the
user's question relates to that report.

Base your answer on the report content and do not
invent values or findings that are not present in it.

MEDICAL REPORT:
----------------
{report_context}
----------------

USER QUESTION:
{request.query}
"""

        # ----------------------------------------------------
        # Process through multi-agent system
        # ----------------------------------------------------

        response_data = process_query(
            query,
            conversation_history=request.conversation_history,
        )

        response_text = (
            response_data["messages"][-1].content
        )

        # ----------------------------------------------------
        # Keep session alive
        # ----------------------------------------------------

        response.set_cookie(
            key="session_id",
            value=session_id,
        )

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"],
        }

        # ----------------------------------------------------
        # Skin lesion result image
        # ----------------------------------------------------

        if (
            response_data["agent_name"]
            == "SKIN_LESION_AGENT, HUMAN_VALIDATION"
        ):

            segmentation_path = os.path.join(
                SKIN_LESION_OUTPUT,
                "segmentation_plot.png",
            )

            if os.path.exists(
                segmentation_path
            ):

                result["result_image"] = (
                    "/uploads/"
                    "skin_lesion_output/"
                    "segmentation_plot.png"
                )

        # ----------------------------------------------------
        # Tell frontend report context is active
        # ----------------------------------------------------

        if report_context:

            result[
                "report_context_active"
            ] = True

        return result

    except RateLimitError as e:

        import traceback
        traceback.print_exc()

        response.set_cookie(
            key="session_id",
            value=session_id,
        )

        return {
            "status": "error",
            "response": (
                "The AI service has temporarily reached its usage limit. "
                "Please wait a few minutes and try again."
            ),
            "agent": "SYSTEM",
            "error_type": "rate_limit",
        }

    except Exception as e:

        import traceback
        traceback.print_exc()

        return {
            "status": "error",
            "response": (
                "Sorry, something went wrong while processing your request. "
                "Please try again."
            ),
            "agent": "SYSTEM",
            "error_type": "internal_error",
        }


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...),
    text: str = Form(""),
):
    """
    Handle medical image uploads.

    IMPORTANT:
    The trained chest X-ray model is currently unavailable
    because its model weights are not included.

    Therefore this endpoint DOES NOT generate a fake diagnosis.

    The image is saved successfully and the frontend receives
    a clear message that X-ray analysis is unavailable.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not image.filename:

        raise HTTPException(
            status_code=400,
            detail="No image selected.",
        )

    filename = image.filename

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in {
        "png",
        "jpg",
        "jpeg",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Use PNG, JPG or JPEG."
            ),
        )

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_data = await image.read()

        if not image_data:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        safe_name = secure_filename(
            filename
        )

        if not safe_name:

            safe_name = (
                f"medical_image.{extension}"
            )

        unique_name = (
            f"{uuid.uuid4()}_{safe_name}"
        )

        image_path = os.path.join(
            FRONTEND_UPLOAD_FOLDER,
            unique_name,
        )

        with open(
            image_path,
            "wb",
        ) as f:

            f.write(image_data)

        print(
            f"Image uploaded: {image_path}"
        )

        # ----------------------------------------------------
        # NO FAKE X-RAY RESULT
        # ----------------------------------------------------

        return {
            "status": "unavailable",
            "agent": "CHEST_XRAY_AGENT",
            "response": """
### Chest X-Ray Analysis

The chest X-ray was uploaded successfully.

However, the trained chest X-ray analysis model
is currently unavailable because its model weights
are not included in this installation.

**No medical prediction was generated.**

Please consult a qualified healthcare professional
for interpretation of the uploaded X-ray.
""",
            "image_uploaded": True,
            "image_path": (
                f"/uploads/frontend/"
                f"{unique_name}"
            ),
        }

    except HTTPException:

        raise

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error processing image: {str(e)}"
            ),
        )


# ============================================================
# MEDICAL PDF REPORT UPLOAD
# ============================================================

@app.post("/upload-report")
async def upload_medical_report(
    response: Response,
    report: UploadFile = File(...),
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None),
):
    """
    Upload and parse a PDF medical report.

    The extracted report text is stored against the browser
    session so later /chat requests can answer questions
    about the same report.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not report.filename:

        raise HTTPException(
            status_code=400,
            detail="No medical report selected.",
        )

    filename = report.filename

    # --------------------------------------------------------
    # Only PDFs
    # --------------------------------------------------------

    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF medical reports "
                "are supported."
            ),
        )

    # --------------------------------------------------------
    # Create session ID
    # --------------------------------------------------------

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    try:

        os.makedirs(
            MEDICAL_REPORT_FOLDER,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Secure filename
        # ----------------------------------------------------

        safe_name = secure_filename(
            filename
        )

        if not safe_name:

            safe_name = (
                "medical_report.pdf"
            )

        # ----------------------------------------------------
        # Unique filename
        # ----------------------------------------------------

        unique_name = (
            f"{uuid.uuid4()}_{safe_name}"
        )

        report_path = os.path.join(
            MEDICAL_REPORT_FOLDER,
            unique_name,
        )

        # ----------------------------------------------------
        # Read PDF
        # ----------------------------------------------------

        report_content = await report.read()

        if not report_content:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded PDF is empty."
                ),
            )

        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        with open(
            report_path,
            "wb",
        ) as f:

            f.write(report_content)

        print(
            f"Medical report saved: "
            f"{report_path}"
        )

        # ====================================================
        # PARSE PDF WITH DOCLING
        # ====================================================

        parser = MedicalDocParser()

        parsed_document, extracted_images = (
            parser.parse_document(
                document_path=report_path,
                output_dir=MEDICAL_REPORT_FOLDER,
                image_resolution_scale=2.0,
                do_ocr=True,
                do_tables=True,

                # IMPORTANT:
                # Formula enrichment disabled because
                # it caused the transformer tokenizer
                # error in the local installation.
                do_formulas=False,

                do_picture_desc=False,
            )
        )

        print(
            "Medical report parsed successfully."
        )

        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        report_text = ""

        try:

            report_text = (
                parsed_document
                .export_to_markdown()
            )

        except Exception as markdown_error:

            print(
                "Markdown extraction failed:",
                markdown_error,
            )

            try:

                report_text = (
                    parsed_document
                    .export_to_text()
                )

            except Exception as text_error:

                print(
                    "Text extraction failed:",
                    text_error,
                )

                report_text = str(
                    parsed_document
                )

        # ====================================================
        # VALIDATE EXTRACTED TEXT
        # ====================================================

        if (
            not report_text
            or not report_text.strip()
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not extract readable "
                    "text from the PDF."
                ),
            )

        # ====================================================
        # STORE REPORT IN SESSION
        # ====================================================

        REPORT_SESSIONS[
            session_id
        ] = report_text

        print(
            "Medical report stored in session."
        )

        print(
            "Session ID:",
            session_id,
        )

        print(
            "Extracted report characters:",
            len(report_text),
        )

        # ----------------------------------------------------
        # Keep same session cookie
        # ----------------------------------------------------

        response.set_cookie(
            key="session_id",
            value=session_id,
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "status": "success",
            "agent": "MEDICAL_REPORT_AGENT",
            "filename": filename,

            "response": (
                "### Medical Report Uploaded "
                "Successfully\n\n"

                f"**File:** {filename}\n\n"

                "The medical report was successfully "
                "read and parsed.\n\n"

                "The report has been kept in the "
                "current session, so you can now ask "
                "follow-up questions about its contents "
                "without uploading it again."
            ),

            "report_text": report_text,

            "extracted_images": len(
                extracted_images
            ),

            "report_context_active": True,
        }

    except HTTPException:

        raise

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Error processing medical report: "
                f"{str(e)}"
            ),
        )


# ============================================================
# HUMAN VALIDATION
# ============================================================

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None),
):
    """
    Handle human validation for medical AI outputs.
    """

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    try:

        response.set_cookie(
            key="session_id",
            value=session_id,
        )

        validation_query = (
            f"Validation result: "
            f"{validation_result}"
        )

        if comments:

            validation_query += (
                f" Comments: {comments}"
            )

        response_data = process_query(
            validation_query
        )

        if (
            validation_result.lower()
            == "yes"
        ):

            return {
                "status": "validated",

                "message": (
                    "**Output confirmed by "
                    "human validator:**"
                ),

                "response": (
                    response_data[
                        "messages"
                    ][-1].content
                ),
            }

        else:

            return {
                "status": "rejected",

                "comments": comments,

                "message": (
                    "**Output requires further "
                    "review:**"
                ),

                "response": (
                    response_data[
                        "messages"
                    ][-1].content
                ),
            }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
):
    """
    Transcribe speech using ElevenLabs.
    """

    if not audio.filename:

        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "No audio file selected"
                )
            },
        )

    temp_audio = None
    mp3_path = None

    try:

        os.makedirs(
            SPEECH_DIR,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Save temporary WebM
        # ----------------------------------------------------

        temp_audio = (
            f"./{SPEECH_DIR}/"
            f"speech_{uuid.uuid4()}.webm"
        )

        audio_content = await audio.read()

        with open(
            temp_audio,
            "wb",
        ) as f:

            f.write(audio_content)

        file_size = os.path.getsize(
            temp_audio
        )

        print(
            f"Received audio file size: "
            f"{file_size} bytes"
        )

        if file_size == 0:

            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Received empty "
                        "audio file"
                    )
                },
            )

        # ====================================================
        # CONVERT TO MP3
        # ====================================================

        mp3_path = (
            f"./{SPEECH_DIR}/"
            f"speech_{uuid.uuid4()}.mp3"
        )

        try:

            audio_segment = (
                AudioSegment.from_file(
                    temp_audio
                )
            )

            audio_segment.export(
                mp3_path,
                format="mp3",
            )

            mp3_size = os.path.getsize(
                mp3_path
            )

            print(
                f"Converted MP3 file size: "
                f"{mp3_size} bytes"
            )

            with open(
                mp3_path,
                "rb",
            ) as mp3_file:

                audio_data = (
                    mp3_file.read()
                )

            print(
                "Converted audio file into "
                "byte array successfully!"
            )

            # =================================================
            # ELEVENLABS TRANSCRIPTION
            # =================================================

            transcription = (
                client.speech_to_text.convert(
                    file=audio_data,
                    model_id="scribe_v1",
                    tag_audio_events=True,
                    language_code="eng",
                    diarize=True,
                )
            )

            if transcription.text:

                return {
                    "transcript":
                        transcription.text
                }

            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        f"API error: "
                        f"{transcription}"
                    ),

                    "details":
                        transcription.text,
                },
            )

        except Exception as e:

            print(
                "Error processing audio:",
                str(e),
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Error processing audio: "
                        f"{str(e)}"
                    )
                },
            )

        finally:

            # ------------------------------------------------
            # Clean temporary files
            # ------------------------------------------------

            try:

                if (
                    temp_audio
                    and os.path.exists(
                        temp_audio
                    )
                ):

                    os.remove(
                        temp_audio
                    )

                if (
                    mp3_path
                    and os.path.exists(
                        mp3_path
                    )
                ):

                    os.remove(
                        mp3_path
                    )

            except Exception as e:

                print(
                    "Could not delete "
                    "temporary files:",
                    e,
                )

    except Exception as e:

        print(
            "Transcription error:",
            str(e),
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            },
        )


# ============================================================
# GENERATE SPEECH
# ============================================================

@app.post("/generate-speech")
async def generate_speech(
    request: SpeechRequest,
):
    """
    Speech generation is disabled for demo.
    """

    return JSONResponse(
        status_code=200,
        content={
            "message":
                "Speech feature disabled."
        },
    )


# ============================================================
# FILE TOO LARGE HANDLER
# ============================================================

@app.exception_handler(413)
async def request_entity_too_large(
    request,
    exc,
):

    return JSONResponse(
        status_code=413,
        content={
            "status": "error",

            "agent": "System",

            "response": (
                "File too large. Maximum "
                "size allowed: "
                f"{config.api.max_image_upload_size}"
                "MB"
            ),
        },
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
    )