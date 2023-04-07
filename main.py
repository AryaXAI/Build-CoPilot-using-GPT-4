import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.vectorstores.pinecone import Pinecone
import pinecone
import openai


# Set the page title and favicon
st.set_page_config(page_title="AryaXAI Copilot", page_icon="https://uploads-ssl.webflow.com/62bec306e1bec5322b5fe292/62ebff064ecda20efd830e1b_Frame%20152.png")

# Set the meta description and custom meta tags
meta_description = "GPT-4 is proving to be one of the advance LLMs to build conversational systems. We are using GPT-4 and demo how you can build Copilot using GPT-4. In this demo, we are showing an example of how we built AryaXAI Copilot. "
custom_meta_data = {"keywords": "ML Observability, GPT-4, GPT-3.5, LLMs, COpilot, AI Copilot", "author": "AryaXAI"}

meta_tags = f'<meta name="description" content="{meta_description}">'
for key, value in custom_meta_data.items():
    meta_tags += f'<meta name="{key}" content="{value}">'

st.markdown(f'<head>{meta_tags}</head>', unsafe_allow_html=True)


logo_url = "https://uploads-ssl.webflow.com/62bec306e1bec5322b5fe292/62ea465f9b9855c7e840518c_Frame%20121.webp"
hyperlink_url = "https://xai.arya.ai/"
colab_png = "https://3.bp.blogspot.com/-apoBeWFycKQ/XhKB8fEprwI/AAAAAAAACM4/Sl76yzNSNYwlShIBrheDAum8L9qRtWNdgCLcBGAsYHQ/s1600/colab.png"
colab = "https://colab.research.google.com/drive/1URBRSkQOWwB7Y9oQpg6HSA_yWiCRmx52?usp=sharing"
second_image_url = "https://app.aryaxai.com/images/book.svg"
second_hyperlink_url = "https://xai.arya.ai/knowledge-hub"

st.markdown(
    f'<div style="display: flex; justify-content: space-between;">'
    f'<a href="{hyperlink_url}" target="_blank" rel="noopener noreferrer"><img src="{logo_url}" width="95"></a>'
    f'<div style="display: flex;">'
    f'<a href="{second_hyperlink_url}" target="_blank" rel="noopener noreferrer" style="margin-right: 10px;"><img src="{second_image_url}" width="30"></a>'
    f'<a href="{colab}" target="_blank" rel="noopener noreferrer"><img src="{colab_png}" width="75"></a>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)


os.environ['OPENAI_API_KEY'] = 'Your-OpenAI-Key'

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

pinecone.init(api_key="PINECONE-API", environment="us-east1-gcp")
index_name = "INDEX_NAME"
psearch = pinecone.Index(index_name)

docsearch = Pinecone.from_existing_index(index_name, embeddings)

llm = OpenAI(temperature=0.1, model_name="gpt-4", max_tokens=512)


def documents(query):
    # Create an embedding for the input query
    doc = openai.Embedding.create(
        input=[query],
        engine="text-embedding-ada-002",
    )

    # Retrieve the documents from Pinecone
    xq = doc['data'][0]['embedding']

    # Get relevant contexts (including the questions)
    res = psearch.query(xq, top_k=4, include_metadata=True)
    contexts = [x['metadata']['text'] for x in res['matches']]
    sources = [x['metadata']['source'] for x in res['matches']]

    return contexts, sources


def chat_complete(query, system_prompt):
    chat = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    
    answer = chat['choices'][0]['message']['content']
    return answer


# Initialize the app
st.title("AryaXAI Copilot")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

query = st.text_input("Type your question here:")

if query:
    #st.write("Shortlisting the knowledge base:")
    search_summary = (documents(query)[0])
    sources = (documents(query)[1])
    
    # Create system prompt
    prompt_start = f"""Act like a customer success manager for this conversation. You are working for AryaXAI. It is an ML Observability tool. 
    You are a data scientist working as a Customer Support Assistant for AryaXAI. AryaXAI is an ML Observability tool for AI. It does all the most important
    tasks as required by ML Observability tool. Your job is to help users to give information about AryaXAI, help them to deploy AryaXAI successfully. 
    Use the following pieces of context to answer the user's question. 
    Here are the sources from there you need to outline the answer:{search_summary}"""

    # Define the second part of the system template as `prompt_end`
    prompt_end = """
    If you don't know the answer, then ask them to connect to AryaXAI support at support@arya.ai.
    Example of your response should be:
    ```
    The answer is 
    ```
    Begin!
    """

    prompt = prompt_start + prompt_end

    current_prompt=prompt
    
    answer = chat_complete(query, current_prompt)
    
    if st.session_state.conversation_history: 
        previous_query, previous_answer = st.session_state.conversation_history[-1][:2]
        docs = [Document(page_content=previous_query + previous_answer)]
        summ = load_summarize_chain(llm, chain_type="map_reduce")
        chat_summary = summ.run(docs)
        #st.write("Shortlisting the knowledge base:")
        search_summary = documents(query)
        sources = (documents(query)[1])
        prompt_start = f"""Act like a customer success manager for this conversation. You are working for AryaXAI. It is an ML Observability tool. 
        You are a data scientist working as a Customer Support Assistant for AryaXAI. AryaXAI is an ML Observability tool for AI. It does all the most important
        tasks as required by ML Observability tool. Your job is to help users to give information about AryaXAI, help them to deploy AryaXAI successfully. 
        This is a follow-up question and here is the summary of the previous conversation: {chat_summary}"""

        prompt_middle = f"""Only use this to answer the question. 
        If there are no 'contexts', then reply to them to connect to our support@aryaxai.com. 
        
        Here is the conext: {search_summary}"""

        # Define the second part of the system template as `prompt_end`
        prompt_end = """
        Use the chat history and context to find the best answer. 
        Example of your response should be:

        ```
        The answer is foo"""

        new_prompt = prompt_start + prompt_middle + prompt_end
        current_prompt = new_prompt

    else:
        current_prompt = prompt

    answer = chat_complete(query, current_prompt)
    st.session_state.conversation_history.append((query, answer, sources))
    
    # Display the rest of the conversation history
    st.write("Conversation history:")
    for i, (q, a, s) in enumerate(reversed(st.session_state.conversation_history)):
        st.markdown(f"**Q: {q}**")
        st.write(f"A: {a}")
        st.write(f"Reference: {s}")

