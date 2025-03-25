# ai_medical_chatbot

📌 Project Overview
MediBot is an AI-powered chatbot designed to provide users with reliable medical information based on an extensive knowledge base. This chatbot leverages Hugging Face’s Mistral-7B model and FAISS-based vector search to deliver accurate and contextually relevant answers from a medical encyclopedia dataset. It enables users to ask health-related questions and receive responses derived from verified medical sources.

Problem Statement
Access to immediate and reliable medical information is a critical challenge, especially for individuals without direct access to healthcare professionals. Many online sources provide misleading or inaccurate information, which can lead to misdiagnosis, panic, or misinformation

 How It Works
Data Processing 🏗️

The chatbot reads and processes a medical encyclopedia (PDF).

It breaks the content into smaller, meaningful chunks for better retrieval.

Smart Search & AI Responses 🔍

When you ask a question, the bot searches the knowledge base using FAISS (a fast search algorithm).

It pulls the most relevant medical information and passes it to an AI model to generate a human-like response.

Streamlit-Powered Chat UI 💬

A simple chat interface allows users to ask anything related to health and receive instant AI-powered responses.


 #Tech Stack
 
AI Model: Mistral-7B (via Hugging Face) 🧠

Data Processing: PyPDFLoader, LangChain, FAISS

Embeddings: sentence-transformers/all-MiniLM-L6-v2

UI: Streamlit (for the chatbot interface)


📌 create_memory_for_llm.py → Processes the medical encyclopedia, splits text into chunks, and stores it in a searchable AI memory.
📌 connect_memory_with_llm.py → Loads the stored medical knowledge and connects it with the AI chatbot.
📌 medibot.py → The main chatbot interface built with Streamlit, allowing users to interact with MediBot.

🚀 How to Use
1️⃣ Clone the Repository
git clone https://github.com/your-username/medibot.git
cd medibot

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Chatbot
streamlit run medibot.py

4️⃣ Start Chatting! 💬


🔮 Future Improvemen
✅ Add a symptom checker for basic diagnosis.
✅ Train the AI on newer medical sources for even better accuracy.
✅ Deploy an online version for easier access.

⚠️ Disclaimer
MediBot is for informational purposes only and should not replace professional medical advice. Always consult a licensed doctor for any medical concerns.

🤝 Contribute & Connect
Want to improve MediBot? Feel free to fork the repo, submit pull requests, or open issues! Let’s make trusted medical information more accessible for everyone. 🚀
