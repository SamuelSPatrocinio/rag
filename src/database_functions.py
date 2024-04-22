from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import chromadb

def create_http_client(host,port):
    return chromadb.HttpClient(host=host, port=port)

def create_or_get_collection(http_client, collection_name, emb_fn=None):
    try:
        collection = http_client.get_collection(name=collection_name)
    except:
        collection = http_client.create_collection(name=collection_name, embedding_function=emb_fn)
    return collection

def load_pdf_files(data_path):
    return PyPDFDirectoryLoader(data_path).load()

def split_documents_in_chunks(documents,chunk_size=1000,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def load_chunks(texts,collection):
    collection.add(documents=list(map(str, texts)), ids=[str(i) for i in range(len(texts))])

def search_data_by_vector(query_vector,collection):
    res = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        include=['distances','embeddings', 'documents', 'metadatas'],
    )
    return res