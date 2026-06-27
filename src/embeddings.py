from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List , Any , Dict
from config import EMBEDDINGS_MODEL
from logger import log



class EmbeddingPipeline:

    """convert The chunks to embeddings!"""

    def __init__(self , model_name: str =EMBEDDINGS_MODEL or "all-MiniLM-L6-v2" ):
        log.info(f"Building : | {model_name} |")

        self.model = SentenceTransformer(model_name)
        

    def embed_chunks(self , chunks : List[Any] ) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks ]

        log.info(f"Embedding.... | {len(texts)} | Chunks ")

        embeddings = self.model.encode(texts , show_progress_bar= True)

        log.success(f"Embedded: {embeddings.shape} dim")

        return embeddings

    def embed_quary(self , query: str)-> np.ndarray:
        return self.model.encode([query]).astype('float32')

if __name__ == "__main__":
    pass 

    # from data_loader import DataFilesLoader
    # from chunks import GenerateChunks
    # loader = DataFilesLoader()
    # chunks = GenerateChunks().split_documents(loader.dataExtractor("temp_upload"))
    # embeddings = EmbeddingPipeline()
    # print(embeddings.embed_chunks(chunks))

