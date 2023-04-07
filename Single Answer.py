import os
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI


st.set_page_config(page_title="AryaXAI Support Bot")

st.title("AryaXAI Support Bot")

os.environ['OPENAI_API_KEY'] = 'Your-OpenAI-Key'
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

index_name = "INDEX_NAME"
pinecone.init(api_key="PINECONE-API", environment="us-east1-gcp")
docsearch = Pinecone.from_existing_index(index_name, embeddings)

chain = VectorDBQAWithSourcesChain.from_chain_type(
    ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0, max_tokens=256),
    chain_type="stuff", 
    vectorstore=docsearch,
)

system_template="""You are a Customer Support Assistant for AryaXAI. AryaXAI is an ML Observability tool for AI. It does all the most important
tasks as required by ML Observability tool. Your job is to help users to give information about AryaXAI, help them to deploy AryaXAI successfull. 
Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),

]
prompt = ChatPromptTemplate.from_messages(messages)


query = st.text_input("Type your question here:")
if query.strip():
    result = chain({"question": query.strip(), "prompt": prompt}, return_only_outputs=True)
    answer = result['answer']
    sources = result['sources']
    output = f"{answer}\nMore information can be found at: {sources}"
    st.write(output)
