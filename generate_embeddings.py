from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # Changed from PyPDFLoader

# Load the os.txt file instead of PDF files
loader = TextLoader('./os.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


vectorstore = Chroma.from_documents(
    docs, embedding_function, persist_directory="./chroma_db_nccn")

print(vectorstore._collection.count())
