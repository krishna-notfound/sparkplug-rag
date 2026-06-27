from typing import List , Any
import pickle
import faiss
import os
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from data_loader import DataFilesLoader
from chunks import GenerateChunks
from embeddings import EmbeddingPipeline
from logger import log
from config import CHUNK_OVERLAP , CHUNK_SIZE , TOP_RESULT , DOC_PATH





class BM25Store:
    def __init__(self , presist_dir :str , chunks: List[Document]):
        self.presist_dir = presist_dir
        os.makedirs(self.presist_dir , exist_ok=True)
        self.index = None
        self.chunks = chunks
    
    def tokenzie(self , text:str) -> List[str]:
        return text.lower().split()
    
    def build_and_save(self)-> None:
        tokenized_docs: List[List[str]] = [self.tokenzie(doc.page_content) for doc in self.chunks]
        try: 
            self.index = BM25Okapi(tokenized_docs)
            self.save()
        except Exception as e:
            log.error(f"Failed To Build! [BM25Store]")

    def save(self)-> None:
        path = os.path.join(self.presist_dir , "bm25.pkl")
        with open(path , "wb") as f:
            pickle.dump(self.index , f)
    
    def search(self , query:str , top_k: int = TOP_RESULT) -> List[Any]:
        query_token = self.tokenzie(query)

        scores = self.index.get_scores(query_token)

        indices = np.argsort(scores)[::-1][:top_k]

        chunk_len = len(self.chunks)
        
        result = []
        for idx in indices:
            doc = self.chunks[idx] if idx < chunk_len else None
            result.append({
                "index": idx,
                "distance": float(scores[idx]), 
                "document": doc
                })
        return result
    def query(self ,query: str, top_k:int = TOP_RESULT): 
        return self.search(query)

class FaissVectorStore:

    def __init__(self , presist_dir:str , chunks : List[Document]):
        self.persist_dir = presist_dir
        os.makedirs(self.persist_dir , exist_ok= True)
        self.index = None
        self.chunks = chunks
        self.emb = EmbeddingPipeline()

    def build_and_save(self):
        log.info(f"||Building VectorStore Saving Documents Embeddings ||\n")

        try:
            embeddings = self.emb.embed_chunks(self.chunks)
            self.add_embeddings(embeddings)
            self.save()

        except Exception as e:
            log.error(f"[Build Function Error] | {e}")

    def add_embeddings(self , embeddings: np.ndarray):

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)


    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "documents.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        log.success(f"Saved Faiss index and Documents to {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_RESULT) -> List[Any]:
        D, I = self.index.search(query_embedding, top_k)
        results = []
        chunk_len = len(self.chunks)
        for idx, dist in zip(I[0], D[0]):
            doc = self.chunks[idx] if idx < chunk_len else None
            results.append({
                "index": idx,
                "distance": dist, 
                "document": doc
                })
            
        return results

    def query(self, query_text: str):
        log.info(f"Querying vector store for: '{query_text}'")

        query_emb = self.emb.embed_quary(query_text)
        r = self.search(query_emb)
        print(type(r))
        return r





if __name__ == "__main__":
    pass
    '''
    # For Debuggings purpose

    loader = DataFilesLoader()
    documents = loader.dataExtractor(DOC_PATH)

    gen_chunks = GenerateChunks(CHUNK_SIZE , CHUNK_OVERLAP)
    chunks = gen_chunks.split_documents(documents)
    vectorstore = FaissVectorStore("vectorstore" , chunks= chunks)
    bm25 = BM25Store("vectorstore" , chunks= chunks)
    vectorstore.build_and_save()
    bm25.build_and_save()

    print(vectorstore.query("what are the policy of company")[2]["document"].page_content)
    print(bm25.query("what are the policy of company")[2]["document"].page_content)

    '''

        




