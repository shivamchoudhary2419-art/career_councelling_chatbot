import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
site_url = os.getenv("SITE_URL", "http://CCC_2025")  # Default for local testing
site_name = os.getenv("SITE_NAME", "Career Counseling Chatbot")

# Streamlit page configuration
st.set_page_config(page_title="Career Counseling Chatbot", layout="centered")
st.title("🎓 Career Counseling Chatbot")
st.markdown("Get personalized advice on education and career paths! Ask about courses, skills, or job strategies.")

# Check API key
if not openrouter_api_key:
    st.error(
        """
        OPENROUTER_API_KEY not found in .env file. Please follow these steps:
        1. Go to OpenRouter (https://openrouter.ai).
        2. Sign up or log in to get an API key.
        3. Add it to a .env file: OPENROUTER_API_KEY=your_key
        4. Optionally, add SITE_URL and SITE_NAME for rankings.
        5. Restart the app.
        """
    )
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your career counselor powered by DeepSeek. I can help with education plans, career paths, or skill development. What's your goal or question?"
        }
    ]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# Enhanced prompt template
prompt_template = """
You are an expert career counselor specializing in education and career planning for students. Provide a concise (under 500 words), actionable, and encouraging response tailored to the student's query. Include specific skills, courses, certifications, or resources (e.g., Coursera, freeCodeCamp, GitHub). If the query is vague, ask a clarifying question to provide targeted advice. Use the conversation history to maintain context and avoid repetition. Structure responses with bullet points for clarity when listing steps or resources. Ensure answers are practical and student-focused.

Conversation History:
{chat_history}

Student's Question:
{input}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Function to call OpenRouter API
def call_openrouter_api(user_input, chat_history):
    try:
        # Format messages with prompt and history
        messages = [
            {"role": "system", "content": prompt_template.format(chat_history=chat_history, input=user_input)}
        ]
        # API request
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 1000,
            }),
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {str(e)}")
        return f"Error: {str(e)}"

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about education or careers (e.g., 'What skills for data science?')"):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.spinner("Thinking..."):
        try:
            # Get chat history from memory
            chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
            # Call OpenRouter API
            answer = call_openrouter_api(user_input, chat_history)
            if answer.startswith("Error:"):
                raise Exception(answer)
            # Append assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
            # Save to memory
            st.session_state.memory.save_context({"input": user_input}, {"output": answer})
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            error_msg = "Sorry, I couldn't process your request. Try rephrasing your question or check your API key."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)

# Clear chat history
with st.sidebar:
    st.markdown("### Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared! Ask me about your career or education goals."}
        ]
        st.session_state.memory.clear()
        st.rerun()