import os
import truststore

truststore.inject_into_ssl()

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model_name="openai/gpt-4o-mini"

def get_llm():
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model_name,
        temperature=0.7,
        max_tokens=512)


llm=get_llm()
response=llm.invoke("Hello, how are you?")
print(response.content)

def technical_agent(question:str)-> str:
    prompt=f"""
    You are a technical agent that answers questions about programming and software development. 
    Please provide a detailed and accurate answer to the following question:
    
    Question: {question}
    
    Answer:
    """
    response=llm.invoke(prompt)
    return response.content

def biling_agent(question:str)->str:
    prompt=f"""
    You are a billing agent that answers questions about billing and payments. 
    Please provide a detailed and accurate answer to the following question:
    
    Question: {question}
    
    Answer:
    """
    response=llm.invoke(prompt)
    return response.content 

def general_agent(question:str)->str:
    prompt=f"""
    You are a general agent that answers questions about various topics. 
    Please provide a detailed and accurate answer to the following question:
    
    Question: {question}
    
    Answer:
    """
    response=llm.invoke(prompt)
    return response.content 

def reviewer_agent(question:str)->str:
    prompt=f"""
    You are a reviewer agent that provides feedback and reviews on the three agent which we created. 
    Check result if correct or not. If not correct, provide the correct answer.
    
    Question: {question}
    
    Answer:
    """
    response=llm.invoke(prompt)
    return response.content