#!/bin/bash

# Create embeddings
python3 create_embeddings.py
# Start chainlit app
chainlit run app.py -h