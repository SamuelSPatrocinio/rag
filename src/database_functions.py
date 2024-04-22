from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from chromadb.config import Settings
import chromadb

def load_pdf_files(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    return documents

def split_documents_in_chunks(documents,chunk_size=1000,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts=text_splitter.split_documents(documents)
    return texts

def load_chunks_to_chroma(texts,collection):
    collection.add(documents=list(map(str, texts)), ids=[str(i) for i in range(len(texts))])

def create_http_client():
    client = chromadb.HttpClient(host='chromadb-service.chromadb.svc.cluster.local', port=8000, settings=Settings(allow_reset=True))
    return client

def create_collection(http_client, collection_name, emb_fn):
    collection = http_client.create_collection(name=collection_name, embedding_function=emb_fn)
    return collection

def search_data_by_vector(query_vector,collection):
    res = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        include=['distances','embeddings', 'documents', 'metadatas'],
    )
    return res