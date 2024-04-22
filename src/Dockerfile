FROM webera/python

RUN pip install --upgrade pip && \
    pip install chainlit && \
    pip install langchain && \
    pip install pypdf && \
    pip install gpt4all && \
    pip install chromadb

WORKDIR /app

COPY . .

ENTRYPOINT [ "chainlit", "run", "app.py", "-h" ]
