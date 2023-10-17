import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent

from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate

langchain.debug = True
load_dotenv('envi.env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
dburl = "sqlite:///./sqlite/moonlay_info.db"


# Initialize Flask app
app = Flask(__name__)

llm = ChatOpenAI(temperature=0, verbose=True)

# Connect to SqlLite Server
db = SQLDatabase.from_uri(dburl)


TEMPLATE = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today". 
Be carefull, question may not use full value of employee id, name, client and description use LIKE instead of EXACT compare
When you need to filter the query by specific column, make sure you also add that column as the select statement

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:

{table_info}

Question: {input}
SQLQuery:
"""

# chains to sqlite database
query_chain = create_sql_query_chain(llm , db, prompt=PromptTemplate.from_template(TEMPLATE))

# Import Knowledge base of information in directory
# Load information from directory
directory = './data/text_profile.txt'
loader = TextLoader(directory)
# Create index
index_creator = VectorstoreIndexCreator()
index = index_creator.from_loaders([loader])
# Create retrieved index
retriever = index.vectorstore.as_retriever()

# System generated prompting
system_message = """Use the following information from below to answer any questions:
Source 1: Employee Information (SQL Database)
You can use this data to answer questions related to employees, their activities, and project details.
<source1>
{source1}
</source1>

Source 2: general information about Moonlay Technologies.
<source2>
{source2}
</source2>
"""
prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])
chat_history = []

# # Combine chain
# full_chain = {
#     "source1": {"question": lambda x: x["question"]} | query_chain | db.run,
#     "source2": (lambda x: x['question']) | retriever,
#     "question": lambda x: x['question'],
# } | prompt | ChatOpenAI()

# Chats!
# response = full_chain.invoke({"question":"what is astrid email?"})
# print(response)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']

        # conversation_history.append({"question": question})
        # chat_input = conversation_history.copy()

        response = full_chain.invoke({"question": question, "chat_history": chat_history})

        # Convert the AIMessage object to a dictionary
        response_dict = {
            "response": response.content  
        }
        return jsonify(response_dict)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':

    # Create the full chat chain
    full_chain = {
        "source1": {"question": lambda x: x["question"]} | query_chain | db.run,
        "source2": (lambda x: x['question']) | retriever,
        "question": lambda x: x['question'],
    } | prompt | ChatOpenAI()

    # Run the Flask app
    app.run()