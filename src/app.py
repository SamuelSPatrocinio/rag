import chainlit as cl
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from database_functions import load_pdf_files, split_documents_in_chunks, create_http_client, create_collection, load_chunks_to_chroma, search_data_by_vector
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

@cl.on_chat_start
async def on_chat_start():
    db_client = create_http_client()
    collection_name = "my-collection"
    try:
        collection = db_client.get_collection(name=collection_name)
    except:
        await cl.Message(f"Could not find collection {collection_name}").send()
        collection = create_collection(db_client, collection_name, OllamaEmbeddings(model="all-minilm",
                   base_url="http://10.50.0.11:11434"))
    documents = load_pdf_files("/app/documents")
    texts = split_documents_in_chunks(documents)
    load_chunks_to_chroma(texts, collection)

    cl.user_session.set("collection", collection)

    await cl.Message("All set").send()

    model = Ollama(model="mistral:v0.2",
                   base_url="http://10.50.0.11:11434")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You must answer the human question using the provided context.",
            ),
            ("human", "question: {question}, context: {context}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    collection = cl.user_session.get("collection")

    msg = cl.Message(content="")

    embeddings = OllamaEmbeddings(model="all-minilm",
                   base_url="http://10.50.0.11:11434")

    msg_embedded = embeddings.embed_query(message.content)

    answer = search_data_by_vector(msg_embedded, collection)

    context = answer['documents'][0][0]

    question = {"question": message.content,
                "context": context}

    runnable_config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

    async for chunk in runnable.astream(question,config=runnable_config):
        await msg.stream_token(chunk)
    await msg.send()

#    await cl.Message(f"{answer['documents'][0][0]}").send()
