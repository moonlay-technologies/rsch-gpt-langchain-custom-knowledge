import os
import sys

import openai
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"

pinecone.init(
    api_key='4b1e03c5-4927-4a1f-8b55-13f985a8c167',
    environment='gcp-starter'
    )
index_name = "openai-index"

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# Load data
directory = 'data'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)

def split_docs(documents,chunk_size=2000,chunk_overlap=0):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
# Embeddings
embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model_name="text-davinci-002")
# Load data to pinecone vector
index = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.as_retriever(search_type="mmr"),
)

chat_history = []
while True:
  if not query:
    query = input("User: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None