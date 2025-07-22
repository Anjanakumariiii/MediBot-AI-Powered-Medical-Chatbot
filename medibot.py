# import os
# import streamlit as st

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_groq import ChatGroq


# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


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
        
#         #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
#         #HF_TOKEN=os.environ.get("HF_TOKEN")  

#         #TODO: Create a Groq API key and add it to .env file
        
#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=ChatGroq(
#                     model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
#                     temperature=0.0,
#                     groq_api_key=os.environ["GROQ_API_KEY"],
#                 ),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             # response=qa_chain.invoke({'query':prompt})

#             # result=response["result"]
#             # source_documents=response["source_documents"]
#             # result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             # #response="Hi, I am MediBot!"
#             # st.chat_message('assistant').markdown(result_to_show)
#             # st.session_state.messages.append({'role':'assistant', 'content': result_to_show})
            
#             response = qa_chain.invoke({'query': prompt})



#             result = response["result"]
#             source_documents = response["source_documents"]

#             # Format source documents
#             sources = "\n\n".join(
#                 f"📄 **Page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))}**\n> {doc.page_content.strip()}"
#                 for doc in source_documents
#             )

#              # Display assistant response with better formatting
#             st.chat_message("assistant").markdown(f"💬 **Answer:**\n\n{result.strip()}")

#             # Expandable section for sources
#             with st.expander("📚 Source Documents"):
#                 st.markdown(sources)

#             # Update session state
#             st.session_state.messages.append({
#                 'role': 'assistant',
#                 'content': f"💬 **Answer:**\n\n{result.strip()}\n\n📚 **Sources:**\n{sources}"
#             })

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()
# import os
# import streamlit as st

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_groq import ChatGroq

# # ✅ Load environment variables only if not using Streamlit Cloud
# if "GROQ_API_KEY" not in st.secrets:
#     from dotenv import load_dotenv, find_dotenv
#     load_dotenv(find_dotenv())

# # ✅ Safely fetch keys from secrets or environment
# groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
# hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, hf_token):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "token": hf_token,
#             "max_length": "512"
#         }
#     )
#     return llm

# def main():
#     st.title("💬 MediBot: Your Medical Assistant")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt = st.chat_input("👨‍⚕️ Ask your medical question here...")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer the user's question.
#         If you don’t know the answer, just say that you don’t know – do not make up an answer.
#         Don't provide anything outside the given context.

#         Context: {context}
#         Question: {question}

#         Start the answer directly, no small talk.
#         """

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("❌ Failed to load the vector store")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=ChatGroq(
#                     model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # Free, Groq-hosted model
#                     temperature=0.0,
#                     groq_api_key=groq_api_key,
#                 ),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response = qa_chain.invoke({'query': prompt})

#             result = response["result"]
#             source_documents = response["source_documents"]

#             # Optional: Highlight medical terms
#             result = result.replace("dehydration", "💧 **dehydration**")
#             result = result.replace("fever", "🌡️ **fever**")
#             result = result.replace("sweat", "💦 **sweat**")
#             result = result.replace("blood vessels", "🩸 **blood vessels**")

#             # Format source documents
#             sources = "\n\n".join(
#                 f"📄 **Page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))}**\n> {doc.page_content.strip()}"
#                 for doc in source_documents
#             )

#             # Display answer
#             st.chat_message("assistant").markdown(f"💬 **Answer:**\n\n{result.strip()}")

#             # Show source docs (collapsible)
#             with st.expander("📚 Source Documents"):
#                 st.markdown(sources)

#             # Save to session
#             st.session_state.messages.append({
#                 'role': 'assistant',
#                 'content': f"💬 **Answer:**\n\n{result.strip()}\n\n📚 **Sources:**\n{sources}"
#             })

#         except Exception as e:
#             st.error(f"⚠️ Error: {str(e)}")

# if __name__ == "__main__":
#     main()
import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ------------------------------
# 🌍 Load API Keys Securely
# ------------------------------
if not st.secrets.get("GROQ_API_KEY"):
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))

# ------------------------------
# 📚 Load Vector Store
# ------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource

def get_vectorstore():
    """Load FAISS vector store with HuggingFace embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ------------------------------
# 🧠 Custom Prompt Template
# ------------------------------
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ------------------------------
# 🤖 Main Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="MediBot - AI Medical Assistant", layout="wide")

    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(to right, #fdfbfb, #ebedee);
    }
    .stChatMessage {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    input[type="text"] {
        border: 2px solid;
        border-image-slice: 1;
        border-width: 2px;
        border-image-source: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        transition: box-shadow 0.3s ease-in-out;
    }
    input[type="text"]:focus {
        box-shadow: 0 0 10px rgba(106, 17, 203, 0.5);
        outline: none;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🩺 MediBot")
    st.caption("Your AI-powered medical assistant – accurate, fast, and context-aware.")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display message history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Input from user
    prompt = st.chat_input("💬 Ask a medical question (e.g., symptoms, medications, or conditions)...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # 🧾 Custom Prompt Instruction
        CUSTOM_PROMPT_TEMPLATE = """
        Answer the question using only the information provided in the context.
        If the answer is not found in the context, respond with “I don’t know.”
        Avoid speculation or unnecessary details.

        Context: {context}
        Question: {question}
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")

            # 🔍 RetrievalQA chain with Groq-hosted model
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=groq_api_key,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # 🔍 Highlight key medical terms
            for term in ["dehydration", "fever", "sweat", "blood vessels"]:
                result = result.replace(term, f"{term}")

            # 📄 Display assistant message
            st.chat_message("assistant").markdown(f"💬 *Answer:*\n\n{result.strip()}")

            # 📚 Display sources
            if source_documents:
                with st.expander("📚 Source Documents"):
                    sources = "\n\n".join(
                        f"📄 *Page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))}*\n> {doc.page_content.strip()}"
                        for doc in source_documents
                    )
                    st.markdown(sources)
            else:
                st.info("No source documents available for this query.")

            # 💾 Save assistant message
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"💬 *Answer:*\n\n{result.strip()}"
            })

        except Exception as e:
            st.error(f"⚠ Error: {str(e)}")

# ------------------------------
# 🚀 Run App
# ------------------------------
if __name__ == "_main_":
    main()
