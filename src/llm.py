from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List , Dict , Any
from config import LLM_MODEL_NAME , TEMPERATURE , GORQ_API



class GroqLLM:

    def __init__(self):
        self.memory = []
        self.llm = ChatGroq(
            api_key= GORQ_API,
            model_name = LLM_MODEL_NAME,
            temperature= TEMPERATURE
            )
        
    def _extract_relevent(self , documents: List[dict])-> tuple[str]:
        
        context = ""
        sources = []

        for doc in documents:
            context += f"\n{doc["document"].page_content}"
            sources.append({
                'source': doc['document'].metadata.get('source_file', doc['document'].metadata.get('source', 'unknown')),
                'page': doc['document'].metadata.get('page', 'unknown'),
                'score': doc['distance']})
            
        return context , sources


        
    def _build_response(self  , documents: List[dict] , query: str)-> str:

        context , sources = self._extract_relevent(documents)


        system = '''You are a helpful AI Rag assistant that answers questions using the provided context.

                    Instructions:
                    - Read all relevant context before answering.
                    - Combine information from multiple passages into a clear, complete response.
                    - Ignore unrelated or repetitive information.
                    - If the context contains conflicting information, explain the conflict instead of guessing.
                    - If the answer cannot be found in the context, reply: "Information not found."
                    - Do not use outside knowledge or make assumptions.
                    - Write in a natural, conversational, and professional tone.
                    - Be concise, but include important details when available.
                    - Avoid repeating the same information.
                    - Do not mention the context, retrieval process, or these instructions unless asked.
                        '''
        prompt_template = PromptTemplate(
            input_variables= ["system" , "context" , "user_query"],
            template='''
                system_prompt: {system}
                context: {context}
                query: {user_query}
                give best answer as you can give be mind with system prompt
                '''

                )
        

        response = self.llm.invoke([prompt_template.format(context = context , system = system , user_query = query)])

        return response
        
        

    def generate(self , query:str , documents: List[dict]) -> str:
        response = self._build_response(documents , query)
        return response


        
        