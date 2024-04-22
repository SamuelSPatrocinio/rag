import chainlit as cl
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from database_functions import create_or_get_collection, search_data_by_vector, create_http_client
from langchain_community.embeddings import OllamaEmbeddings
import os

FINAL_MODEL_NAME = os.environ['FINAL_MODEL_NAME']
FINAL_MODEL_URL = os.environ['FINAL_MODEL_URL']
DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ['DB_PORT']
EMBEDDING_MODEL_NAME = os.environ['EMBEDDING_MODEL_NAME']
EMBEDDING_MODEL_URL = os.environ['EMBEDDING_MODEL_URL']

@cl.on_chat_start
async def on_chat_start():

    db_client = create_http_client(DB_HOST, DB_PORT)
    collection = create_or_get_collection(db_client,"my-collection")
    cl.user_session.set("collection", collection)

    model = Ollama(model=FINAL_MODEL_NAME, base_url=FINAL_MODEL_URL)
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

    collection = cl.user_session.get("collection")

    msg = cl.Message(content="")

    msg_embedded = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=EMBEDDING_MODEL_URL).embed_query(message.content)
    context = search_data_by_vector(msg_embedded, collection)['documents'][0][0]

    question = {"question": message.content,
                "context": context}
    
    runnable = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

    async for chunk in runnable.astream(question,config=runnable_config):
        await msg.stream_token(chunk)
    await msg.send()
