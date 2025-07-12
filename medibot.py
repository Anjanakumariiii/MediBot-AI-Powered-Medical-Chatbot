# import os
# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint



# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#          llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,"max_length":"512"})
#          return llm    

# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

    
#     prompt=st.chat_input("Pass your prompt here")
    
#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})
        

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """

#         HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN=os.environ.get("HF_TOKEN")

#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")
            

#                 qa_chain=RetrievalQA.from_chain_type(
#                      llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                      chain_type="stuff",
#                      retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                      return_source_documents=True,
#                     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#                 )

#                 response=qa_chain.invoke({'query':prompt})

#                 result=response["result"]
#                 source_documents=response["source_documents"]
#                 result_to_show=result+"\nSource Docs:\n"+str(source_documents)

#                 # response="hi, i am anjanabot!"
#                 st.chat_message('assistant').markdown(response)
#                 st.session_state.messages.append({'role':'assistant', 'content': response})
#         except Exception as e:
#                 st.error(f"Error: {str(e)}")

# if __name__=="__main__":
#     main()        
# import os
# import streamlit as st

# from langchain_community.embeddings import HuggingFaceEmbeddings

# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceEndpoint

# # from langchain_huggingface import HuggingFaceEndpoint

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt=st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """
        
#         HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN=os.environ.get("HF_TOKEN")

#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain=RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response=qa_chain.invoke({'query':prompt})

#             result=response["result"]
#             source_documents=response["source_documents"]
#             result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             #response="Hi, I am MediBot!"
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

# Page config
st.set_page_config(page_title="MediBot AI 🩺", page_icon="🩺", layout="wide")

# Modern CSS Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #F9F9F9, #EEF2F3);
        padding: 2rem;
    }
    .title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.2rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .user-msg {
        background-color: #D6EAF8;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .bot-msg {
        background-color: #FADBD8;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #95A5A6;
        margin-top: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title & subtitle
st.markdown("<div class='title'>🩺 MediBot AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your Personal AI Medical Assistant</div>", unsafe_allow_html=True)

# Load Vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

# Main Chatbot Logic
def main():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role_style = 'user-msg' if msg['role'] == 'user' else 'bot-msg'
        st.markdown(f"<div class='{role_style}'>{msg['content']}</div>", unsafe_allow_html=True)

    user_prompt = st.chat_input("Type your medical query here...")

    if user_prompt:
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say you don't know — don't make things up.
        Stick strictly to the given context.

        Context: {context}
        Question: {question}

        Directly answer without small talk.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        # HF_TOKEN = os.environ.get("HF_TOKEN")
        HF_TOKEN = st.secrets["HF_TOKEN"]


        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vector database failed to load.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': user_prompt})
            result = response["result"]
            sources = response["source_documents"]

            result_to_show = result + "\n\n**📚 Sources:**\n"
            for idx, doc in enumerate(sources, 1):
                result_to_show += f"- 📄 Document {idx}: {doc.metadata.get('source', 'Unknown')}\n"

            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("<div class='footer'>© 2025 MediBot AI | Built with ❤️ by Anjana Kumari</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
