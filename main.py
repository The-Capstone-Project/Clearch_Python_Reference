import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain


def get_relevent_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn",
                       embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context


context = get_relevent_context_from_db(
    "how much memory do i have left in my system")

print(context)
