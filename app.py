import sys
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

# Ensure necessary libraries are installed
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

# Initialize Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    groq_api_key="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"
)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# Function to query AI model
def query_llama3(user_query):
    system_prompt = "System Prompt: Your AI clone personality based on Daksha Patil."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    try:
        response = chat.invoke(messages)
        st.session_state.memory.append({"input": user_query, "output": response.content})
        return response.content
    except Exception as e:
        return f"⚠ API Error: {str(e)}"

# Streamlit App
def main():
    # Custom CSS for Styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
        }

        .main {
            background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .stTextInput > div > div > input {
            border: 2px solid #6366f1;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 10px rgba(79, 70, 229, 0.2);
        }

        .stButton > button {
            background-color: #4f46e5;
            color: white;
            border-radius: 25px;
            padding: 10px 30px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #4338ca;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4);
        }

        .chat-container {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            max-height: 500px;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .user-message {
            background: linear-gradient(135deg, #6366f1 0%, #a5b4fc 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 15px 15px 0 15px;
            margin: 10px 0;
            max-width: 70%;
            animation: fadeIn 0.5s ease-in-out;
            transition: transform 0.2s ease;
        }

        .user-message:hover {
            transform: scale(1.02);
        }

        .ai-message {
            background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
            color: #1f2937;
            padding: 12px 20px;
            border-radius: 15px 15px 15px 0;
            margin: 10px 0;
            max-width: 70%;
            animation: fadeIn 0.5s ease-in-out;
            transition: transform 0.2s ease;
        }

        .ai-message:hover {
            transform: scale(1.02);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .sidebar .sidebar-content {
            background-color: #1f2937;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }

        h1 {
            color: #1f2937;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            margin-top: 20px;
            font-size: 14px;
        }

        .stSpinner {
            color: #4f46e5;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for Chat Management
    with st.sidebar:
        st.markdown("### Chat Controls")
        st.markdown("Manage your chat experience.")
        if st.button("Clear Chat History", key="clear_history"):
            st.session_state.memory = []
            st.experimental_rerun()

    # Main App Layout
    st.markdown("<h1>🤖 AI Chatbot by Daksha Patil</h1>", unsafe_allow_html=True)
    st.markdown("Welcome to your professional AI assistant. Start chatting below!")

    # Initialize session memory
    if "memory" not in st.session_state:
        st.session_state.memory = []

    # Chat History Display
    st.markdown("### Conversation")
    with st.container():
        if st.session_state.memory:
            for chat in st.session_state.memory:
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-start;'>
                        <div class='user-message'><strong>You:</strong> {chat['input']}</div>
                    </div>
                    <div style='display: flex; justify-content: flex-end;'>
                        <div class='ai-message'><strong>AI:</strong> {chat['output']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<div style='color: #6b7280; text-align: center;'>No previous chat history. Start a conversation!</div>", unsafe_allow_html=True)

    # User Input Form
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question:", placeholder="Type your message here...")
        submit = st.form_submit_button("Send")
        if submit and user_query:
            with st.spinner("AI is thinking..."):
                response = query_llama3(user_query)
            st.experimental_rerun()

    # Footer
    st.markdown("<div class='footer'>Built with ❤️ by Daksha Patil | Powered by Groq & Streamlit</div>", unsafe_allow_html=True)

# ✅ Correct entry point
if __name__ == "__main__":
    main()