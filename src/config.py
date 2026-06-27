from dotenv import load_dotenv
import os

load_dotenv()


GORQ_API = os.getenv("GORQ_API")



EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_RESULT = 10
FINAL_RESULT_RRF = 5
PRESIST_DIR = "vectorstore"
DOC_PATH = "temp_upload"


LLM_MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.8





