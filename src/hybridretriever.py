from typing import List , Any
from collections import defaultdict
from langchain_core.documents import Document
import os
from logger import log
from basestore import FaissVectorStore , BM25Store
from config import TOP_RESULT, FINAL_RESULT_RRF

from llm import GroqLLM
from data_loader import DataFilesLoader
from chunks import GenerateChunks
from embeddings import EmbeddingPipeline
from config import CHUNK_OVERLAP , CHUNK_SIZE , TOP_RESULT , DOC_PATH

class HybridRetiever:
    def __init__(self,vectorstore : FaissVectorStore , bm25 : BM25Store , rrf_k : int = 60):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.rrf_k = rrf_k

        
        self.vectorstore.build_and_save()
        self.bm25.build_and_save()

    def _rrf(self , *result_list):

        rrf_score = defaultdict(float)
        chunk_lookup = {}

        for results in result_list:
            for rank , result in enumerate(results):
                # print(result)
                # print(type(result))
                chunk_id = result["document"].metadata["chunk_id"]
                chunk_lookup[chunk_id] = result

                rrf_score[chunk_id] = (1 / (self.rrf_k + rank +1))

        ranked_chunks = sorted(
            rrf_score.items(),
            key= lambda tup:tup[1],
            reverse=True
        )

        marge_result = []

        for chunk_id , score in ranked_chunks:
            result = chunk_lookup[chunk_id].copy()
            result["rrf_score"] = score
            marge_result.append(result)
        # print(marge_result)

        return marge_result


    def query(self , query:str , final_k : int = FINAL_RESULT_RRF):
        try:
            vs_result : List[dict] = self.vectorstore.query(query)
            bm25_result : List[dict] = self.bm25.query(query)
            # print(vs_result)


            fused_result = self._rrf(vs_result,bm25_result)

            return fused_result[:final_k]


        except Exception as e:
            log.error(f"[Failed in HybridRetiever.quary()] {e} ")


if __name__ == "__main__":

    '''
    
    loader = DataFilesLoader()
    documents = loader.dataExtractor(DOC_PATH)

    gen_chunks = GenerateChunks(CHUNK_SIZE , CHUNK_OVERLAP)
    chunks = gen_chunks.split_documents(documents)
    vectorstore = FaissVectorStore("vectorstore" , chunks= chunks)
    bm25 = BM25Store("vectorstore" , chunks= chunks)

    query = "explain about Ritika Malhotra how many leave she take is see good accounding to the perform or she need to layoff"
    retiever = HybridRetiever(vectorstore ,bm25)

    llm = GroqLLM()
    llm.generate(
        query= query,
        documents= retiever.query(query)
    )
    '''
    

