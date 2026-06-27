from pathlib import Path
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from typing import Dict, List , Any
import os
from logger import log

class DataFilesLoader:

    """Handles Serveral Type of files and convert Them in Documents for chunking for chunking to be applied!
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """

    ## root/per_session_dir/**.[PDF, TXT, CSV, Excel, Word, JSON]
    def __init__(self , rootdir:str = "rootdata/"):

        self.rootdir = rootdir
        os.makedirs(self.rootdir , exist_ok= True)


    def dataExtractor(self , dir:str ) -> List[Document]:
        all_documents = []
        data_path = Path(self.rootdir+dir).resolve()
        log.debug(f"| {data_path} | Used for Readings Files!")

        loader = {
            "*.pdf": PyPDFLoader,
            "*.txt": TextLoader,
            "*.csv": CSVLoader,
            "*.xlsx": UnstructuredExcelLoader,
            "*.docx": Docx2txtLoader,
            "*.json":JSONLoader,

        }

        for pattern , Loader in loader.items():
            
            files = list(data_path.glob(f"**/{pattern}"))
            extension = pattern.replace("*", "")
            if len(files) >0: 
                log.info(f"Found {len(files)} {extension} files:\n\n")
                log.success([str(f).split("\\")[-1] for f in files])

            for file in files:
                
                print(f"[DEBUG] Loading File {file}")
                try:
                    loader = Loader(str(file))
                    documents = loader.load() 
                    # print(documents)

                    log.success(f"Loaded {len(documents)} documents from {file.name}")

                    

                    for doc in documents:
                        doc.metadata['source_file'] = file.name
                        doc.metadata['file_type'] = extension
                        doc.metadata["document_id"] = f"doc_{uuid.uuid1().hex[:8]}"

                except Exception as e:
                    print(f"Failed To Load Documents! {e}")

                
                all_documents.extend(documents)
        print(f"\n\nTotal documents loaded: {len(all_documents)}\n")
        return all_documents
    

if __name__ == "__main__":
   pass











