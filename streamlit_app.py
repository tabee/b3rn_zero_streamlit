import streamlit as st
from langchain.llms import OpenAI
from app.question_optimizer_chain import optimize_question
from app.conversational_retrieval_agent import ask_agent


sys_path_local = '/workspaces/b3rn_zero_streamlit'
sys_path_streamlit = '/mount/src/b3rn_zero_streamlit/'
sys_path = sys_path_streamlit

st.title('🤖 B3rn Zero Chat')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')


def generate_response(input_text, openai_api_key=openai_api_key):
    optimized_question = optimize_question(
        input_text, openai_api_key=openai_api_key, sys_path=sys_path)
    st.info(optimized_question)
    answer = ask_agent(optimized_question,
                       openai_api_key=openai_api_key, sys_path=sys_path)
    st.info(answer['output'])


with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text, openai_api_key=openai_api_key)
