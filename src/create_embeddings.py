from database_functions import *
import os
from langchain_community.embeddings import OllamaEmbeddings

DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ['DB_PORT']
EMBEDDING_MODEL_NAME = os.environ['EMBEDDING_MODEL_NAME']
EMBEDDING_MODEL_URL = os.environ['EMBEDDING_MODEL_URL']
DOCUMENTS_PATH = os.environ['DOCUMENTS_PATH']


def main():
    # Split
    pdf_files = load_pdf_files(DOCUMENTS_PATH)
    chunks = split_documents_in_chunks(pdf_files,chunk_size=1000,chunk_overlap=50)

    # Load
    db_client = create_http_client(DB_HOST, DB_PORT)
    collection = create_or_get_collection(db_client, "my-collection", OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=EMBEDDING_MODEL_URL))
    load_chunks(chunks, collection)
    

if __name__ == "__main__":
    main()