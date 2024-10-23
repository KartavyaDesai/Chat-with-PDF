import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# Explainer: Load environment variables, including API keys.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT") #You can name it anything for purpose of a project. It will be reflected in your tracker.

#Using Hugging Face Embeddings to convert words to vectors.
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Explainer: Set up the main UI of the Streamlit app.
st.title("Chat With PDF")
api_key = st.text_input("Drop your Groq API key here", type="password")

if api_key:
    # Explainer: Initialize language model (LLM) using the provided API key.
    llm = ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
    # Explainer: Create a session ID for chat management.
    session_id = st.text_input("Define a session key", value="435589703111")

    # Explainer: Initialize session state for storing conversation data.
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Upload PDF file(s)", type='pdf', accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        # Explainer: Process each uploaded PDF file and extract text.
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            # Function: Load and extract text content from the uploaded PDF.
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Explainer: Split extracted text into manageable chunks for embedding and retrieval.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        final_documents=text_splitter.split_documents(documents)

        # Function: Create FAISS vector store from the text chunks, enabling efficient text retrieval.
        vectors=FAISS.from_documents(documents=final_documents,embedding=embeddings)
        retriever = vectors.as_retriever()
    
        contextual_prompt = (
            "Given a chat history and the latest user question" 
            "which might reference context in the chat history, " 
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question"
            "just reformulate it if needed and otherwise return it as is."
        )

        #MessagesPlaceholder("chat_history") is used to insert a placeholder within a prompt template to dynamically include chat history during the generation process.
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextual_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        # Function: Create a retriever that can take chat history into account when retrieving information.
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_prompt)
        
        system_prompt = ( 
                        "You are an assistant for question-answering tasks. " 
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the " 
                        "answer concise."
                        "\n\n"
                        "{context}" 
                        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        # Function: Combine the retriever with the question-answering chain to create a retrieval-augmented generation (RAG) chain.
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Explainer: Retrieve chat history for a specific session.
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        # Function: Create a runnable object that integrates chat history for a more contextual chat experience.
        conversational_rag = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Explainer: Capture user input and generate responses based on retrieved information.
        user_input = st.text_input("Ask a question : ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Enter API Key")
