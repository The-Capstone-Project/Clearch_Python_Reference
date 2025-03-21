import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


def get_relevent_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn",
                       embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=5)
    for result in search_results:
        context += result.page_content + "\n"
    return context


system_prompt_template = f"""
You are Clearch, a specialized virtual assistant for terminal environments that provides system information.

ROLE:
- Help users access hardware and software information about their system
- Troubleshoot system issues and answer technical questions
- Provide guidance on system operations, installations, and configurations

INSTRUCTIONS:
- Use the provided system context to answer questions accurately
- For memory, disk space, or hardware queries, extract precise information from context
- When troubleshooting, provide step-by-step solutions based on the system specs
- For installation requests, verify compatibility with the system before giving instructions
- If information is insufficient, request specific details to provide better assistance

RESPONSE STYLE:
- Keep responses concise and technically accurate
- Use formatting for readability when displaying system data
- Avoid speculation when system information is incomplete
- Prioritize practical solutions over theoretical explanations

Remember to base all answers solely on the actual system information provided in the context.
"""


def get_response(query):
    context = get_relevent_context_from_db(query)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable")

    model = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )

    human_message = """
    <CONTEXT>
        {context}
    </CONTEXT>

    <QUESTION>
    {query}
    </QUESTION>
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", human_message)
    ])

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "query": query
    })

    return response.content


if __name__ == "__main__":
    query = "do i have docker installed in this system?"
    response = get_response(query)
    print(response)
