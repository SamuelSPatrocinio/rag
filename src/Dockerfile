FROM webera/python

RUN pip install --upgrade pip && \
    pip install chainlit && \
    pip install langchain && \
    pip install pypdf && \
    pip install chromadb

WORKDIR /app

COPY . .

ENTRYPOINT ["sh", "start.sh"]
