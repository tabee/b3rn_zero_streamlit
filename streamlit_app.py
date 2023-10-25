""" Streamlit app for B3rn Zero Chat. """
import streamlit as st
from app.question_optimizer_chain import optimize_question
from app.conversational_retrieval_agent import ask_agent__chch, ask_agent__eak


SYS_PATH_LOCAL = '/workspaces/b3rn_zero_streamlit'
SYS_PATH_STREAMLIT = '/app/b3rn_zero_streamlit/'
SYS_PATH = SYS_PATH_STREAMLIT

st.title('🧙‍♂️ B3rn Zero Chat 0.0.2')

# Konfigurationsbereich in der Sidebar
openai_api_key = st.sidebar.text_input(
    'OpenAI API Key',
    value='',
    type='password')
optimize_input = st.sidebar.toggle('Optimizing question')
ask_eak = st.sidebar.toggle('Ask 👴', value=True)
ask_ch = st.sidebar.toggle('Ask 🍫', value=True)

def generate_response(input_text):
    """ Generate response from input text. """
    if optimize_input:
        input_text = optimize_question(
            input_text,
            openai_api_key,
            SYS_PATH)
        with st.chat_message("System", avatar="🧙‍♂️"):
            st.write(f"Optimized Question: {input_text}")

    if ask_ch:
        answer = ask_agent__chch(
            input_text,
            openai_api_key,
            SYS_PATH)
        with st.chat_message("CH", avatar="🍫"):
            st.write(answer['output'])

    if ask_eak:
        answer = ask_agent__eak(
            input_text,
            openai_api_key,
            SYS_PATH)
        with st.chat_message("EAK", avatar="👴"):
            st.write(answer['output'])

# Willkommensnachricht
with st.chat_message("System", avatar="🧙‍♂️"):
    st.write("Welcome to B3rn Zero Chat! 🧙‍♂️")

# Chat Eingabebereich
prompt = st.chat_input("Enter your message")
if prompt:
    with st.chat_message("User"):
        st.write(prompt)

    if not optimize_input:
        with st.chat_message("System"):
            st.write('Optimizing question deactivated.')
    if not openai_api_key.startswith('sk-'):
        with st.chat_message("System"):
            st.write('Please enter your OpenAI API key!')
    if openai_api_key.startswith('sk-'):
        generate_response(prompt)
