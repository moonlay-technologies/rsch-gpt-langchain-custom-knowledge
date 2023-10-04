from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent

from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate

load_dotenv('envi.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create chat agent 
# Create connection to sqlite database
dburl = "sqlite:////...\...\sqlite\moonlay_info.db"

# Connect to SqlLite Server
db = SQLDatabase.from_uri(dburl)
# chains to sqlite database
query_chain = create_sql_query_chain(ChatOpenAI(temperature=0), db)

# Import Knowledge base of information in directory
# Load information from directory
directory = 'data'

loader = DirectoryLoader(directory)
loads = loader.load()
# Create index
index_creator = VectorstoreIndexCreator()
index = index_creator.from_loaders([loader])
# Create retrieved index
retriever = index.vectorstore.as_retriever()

# System generated prompting
system_message = """ Use the following information from below to answer any questions

Source 1: a SQL database that contains data about employee informations
<source1>
{source1}
</source1>

Source 2: a text file containing company informaiton
<source2>
{source2}
</source2>
"""
prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])

# Combine chain
full_chain = {
    "source1": {"question": lambda x: x["question"]} | query_chain | db.run,
    "source2": (lambda x: x['question']) | retriever,
    "question": lambda x: x['question'],
} | prompt | ChatOpenAI()

# Chats!
# response = full_chain.invoke({"question":"what is astrid email?"})
# print(response)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']

        response = full_chain.invoke({"question": question})
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)