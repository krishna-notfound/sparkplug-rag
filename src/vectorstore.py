from typing import List , Any
import pickle
import faiss
import os
import numpy as np

from langchain_core.documents import Document
from data_loader import DataFilesLoader
from chunks import GenerateChunks
from embeddings import EmbeddingPipeline
from logger import log


class FaissVectorStore:

    def __init__(self , presist_dir:str ,  docs_path : str , chunk_size: int = 1000, chunk_overlap: int = 200 ):
        self.persist_dir = presist_dir
        os.makedirs(self.persist_dir , exist_ok= True)
        self.docs_path = docs_path
        self.chunk_overlap = chunk_overlap
        self.chunk_size = chunk_size
        self.index = None
        self.chunked_docs = []
        self.emb = EmbeddingPipeline()

    def build_and_save(self):
        log.info(f"||Building VectorStore Saving Documents Embeddings ||\n")
        

        self.index = None
        self.chunked_docs = []

        try:
            ## Build Ingestion Pipeline
            loader = DataFilesLoader()
            documents = loader.dataExtractor(self.docs_path)

            gen_chunks = GenerateChunks(self.chunk_size , self.chunk_overlap)
            chunks = gen_chunks.split_documents(documents)

            
            embeddings = self.emb.embed_chunks(chunks)

            print("add embeding called!")
            self.add_embeddings(embeddings , chunks)
            print("add save called!")
            self.save()

        except Exception as e:
            log.error(f"[Build Function Error] | {e}")

    def add_embeddings(self , embeddings: np.ndarray , chunked_docs: List[Document]):

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            print(self.index , "selfindex")
        self.index.add(embeddings)
        print(self.index , "selfindex")


        if chunked_docs:
            self.chunked_docs.extend(chunked_docs)

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "documents.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunked_docs, f)
        log.success(f"Saved Faiss index and Documents to {self.persist_dir}")
    
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "documents.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.chunked_docs = pickle.load(f)
        log.success(f"Loaded Faiss index and documents from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Any]:
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            doc = self.chunked_docs[idx] if idx < len(self.chunked_docs) else None
            results.append({"index": idx, "distance": dist, "Data": doc})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")

        query_emb = self.emb.embed_quary(query_text)
        # print(query_emb)
        return self.search(query_emb, top_k=top_k)

if __name__ == "__main__":   
    vectorstore = FaissVectorStore("vectorstore" , "temp_upload")
    vectorstore.build_and_save()


    print(vectorstore.query("what are the policy of company"))



        




