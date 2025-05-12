#Daksha Patil's AI Chatbot

A Streamlit-based web application for an AI chatbot powered by Groq, LangChain, and ChromaDB. This project, created by Daksha Patil, allows you to interact with a personalized AI assistant featuring a sleek UI and persistent conversation history.

#🚀 Getting Started

Follow these steps to set up and run Daksha Patil's AI Chatbot on your local machine.

#Prerequisites
Python: Version 3.8 or higher
pip: Python package manager
Git: For cloning the repository
A Groq API Key: Obtain one from Groq

A modern web browser (e.g., Chrome, Firefox)

#📥 Installation & Setup

#Clone the Repository

git clone https://github.com/Up9777/Ai_ChatBot.git\

cd Ai_ChatBot

#Create and Activate a Virtual Environment (Recommended)

python -m venv venv\

source venv/bin/activate

On Windows, use:

venv\Scripts\activate

This isolates dependencies and prevents conflicts with other projects.

Install Dependencies

Install the required Python packages listed in requirements.txt:

pip install -r requirements.txt

If requirements.txt is missing, you can manually install the core dependencies:

pip install streamlit langchain langchain-huggingface langchain-groq sentence-transformers chromadb

Set Up the Groq API Key

The application requires a Groq API key to function. Set it as an environment variable:

export GROQ_API_KEY="your-groq-api-key"

On Windows, use:

set GROQ_API_KEY=your-groq-api-key

Replace your-groq-api-key with your actual key from Groq. Alternatively, you can hardcode the key in app.py (not recommended for security reasons).

Run the Streamlit App

Launch the application with:

streamlit run app.py

The app will open automatically in your default browser at:

http://localhost:8501

3📁 Project Structure

#Ai_ChatBot/\

│
├── app.py # Main Streamlit application script
├── requirements.txt # List of Python dependencies
├── README.md # Project documentation
├── chroma_db/ # Directory for ChromaDB persistent storage
└── venv/ # Virtual environment (if created)

#🛠️ Built With
Streamlit: Web app framework for the UI
LangChain: For managing LLM interactions
Groq: API for the LLaMA3 model
ChromaDB: Vector database for knowledge storage
Sentence Transformers: For text embeddings
Python: Core programming language

#⚙️ Usage
Interact with the Chatbot: Type your questions in the text input field and click "Send" to get responses from the AI.
Clear Chat History: Use the "Clear Chat History" button in the sidebar to reset the conversation.
Persistent Storage: Conversation data is stored in memory (session state) and embeddings are saved in chroma_db/.

#🛠️ Troubleshooting
ModuleNotFoundError: Ensure all dependencies are installed (pip install -r requirements.txt). If errors persist, verify your Python version (3.8+) and virtual environment.
ChromaDB Errors: If you see SQLite-related issues, ensure your environment supports pysqlite3. Install it manually if needed:
pip install pysqlite3

Groq API Key Issues: Verify your API key is set correctly. If you get authentication errors, check Groq’s documentation.
Port Conflict: If localhost:8501 is in use, Streamlit will prompt you to choose another port.

#🤝 Contributing
Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

#📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

#📬 Contact
For questions or feedback, reach out to Daksha Patil via GitHub Issues or email at daksha.patil@example.com.
