""" Streamlit app for B3rn Zero Chat. """
import streamlit as st
from app.question_optimizer_chain import optimize_question
from app.conversational_retrieval_agent import ask_agent


SYS_PATH_LOCAL = '/workspaces/b3rn_zero_streamlit'
SYS_PATH_STREAMLIT = '/mount/src/b3rn_zero_streamlit/'
SYS_PATH = SYS_PATH_STREAMLIT

st.title('🤖 B3rn Zero Chat 0.0.1')

openai_api_key = st.sidebar.text_input(
    'OpenAI API Key',
    value='',
    type='password')
optimize_input = st.sidebar.toggle('Optimizing question')

def generate_response(input_text):
    """ Generate response from input text. """
    if optimize_input:
        input_text = optimize_question(
            input_text,
            openai_api_key,
            SYS_PATH)
        st.info(input_text)
    answer = ask_agent(
        input_text,
        openai_api_key,
        SYS_PATH)
    st.info(answer['output'])


with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')

    if not optimize_input:
        st.toast('Optimizing question deactivated.')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
