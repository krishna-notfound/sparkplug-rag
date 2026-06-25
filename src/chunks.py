from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List
from logger import log

class GenerateChunks:
    def __init__(self , chunks_size : int , chunks_overlap : int, ):

        self.all_chunks = []
        self.seps = [
        "\n\n","\n"," ",".",",",
        "\u200b",  
        "\uff0c",  
        "\u3001",  
        "\uff0e",  
        "\u3002",  
        "",]
        self.recursive_spliter = RecursiveCharacterTextSplitter(
            chunk_size= chunks_size,
            chunk_overlap = chunks_overlap,
            separators = self.seps,
            length_function = len
        )

    def split_documents(self , documents : List[Document] ) -> List[Document]:
        try:
            chunks = self.recursive_spliter.split_documents(documents)
            

            for idx , chunk in enumerate(chunks):

                chunk.metadata.update({
                "chunk_id": idx,
                "chunk_length": len(chunk.page_content)
                })
            log.success(f"Convert {len(documents)} to {len(chunks)} ")
            return chunks
        except Exception as e:
            log.error(f"| {e}")


        

        
            





    
