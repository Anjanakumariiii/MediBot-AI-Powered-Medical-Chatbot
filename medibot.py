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
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Page config and CSS styling
st.set_page_config(page_title="🩺 MediBot AI", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .reportview-container { padding: 2rem; }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stChatInput textarea {
        font-size: 1rem;
    }
    .footer {
        position: fixed;
        bottom: 5px;
        text-align: center;
        width: 100%;
        color: gray;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Define custom prompt template
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load LLM with correct task
# def load_llm(huggingface_repo_id, HF_TOKEN):
#     return HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         task="text2text-generation",  # KEY FIX: set correct task
#         temperature=0.5,
#         huggingfacehub_api_token=HF_TOKEN
#     )
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text2text-generation",  # <---- ✅ Corrected here
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN
    )


# Main Streamlit app function
def main():
    st.title("🩺 MediBot AI - Your Health Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("💬 Type your health-related question here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the context provided below to answer the user's health-related question.
        If unsure, say you don't know. Stick to the given context.

        Context: {context}
        Question: {question}

        Provide a clear, friendly answer.
        """

        HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"
        HF_TOKEN = st.secrets.get("HF_TOKEN")

        if not HF_TOKEN:
            st.error("⚠️ Hugging Face token is missing! Please set 'HF_TOKEN' in Streamlit Cloud Secrets.")
            return

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            sources = response["source_documents"]

            result_to_show = f"**Answer:**\n{result}\n\n**📚 Sources:**\n"
            for idx, doc in enumerate(sources, 1):
                result_to_show += f"- 📄 Document {idx}: {doc.metadata.get('source', 'Unknown')}\n"

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error("❌ An error occurred while generating the response.")
            st.write(e)

    st.markdown("<div class='footer'>© 2025 MediBot AI | Built with ❤️ by Nainaa</div>", unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    main()
