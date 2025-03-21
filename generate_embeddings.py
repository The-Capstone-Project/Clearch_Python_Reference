from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# * Make the OS Scraper dump the contents of the scrpe to ths os.txt file
# * Put this os.txt file in some global location where the Vector store can access it

loader = TextLoader('./os.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


vectorstore = Chroma.from_documents(
    docs, embedding_function, persist_directory="./chroma_db_nccn")  # Give some global place to dummp the chroma directory

print(vectorstore._collection.count())
