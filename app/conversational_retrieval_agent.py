''' 
https://python.langchain.com/docs/use_cases/
question_answering/how_to/conversational_retrieval_agents  '''
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent, create_retriever_tool)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback


def ask_agent(query, openai_api_key, sys_path, model='gpt-3.5-turbo-16k'):
    '''Display the answer to a question.'''
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    new_db1 = FAISS.load_local(
        f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_4096',
        embeddings)
    # new_db2 = FAISS.load_local(
    #      f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_512',
    #      embeddings)
    
    # new_db3 = FAISS.load_local(f'{sys_path}/data/vectorstores/ch_ch_texts_faiss_index_4096',
    #                          embeddings)

    # new_db1.merge_from(new_db2)
    # new_db1.merge_from(new_db3)
    new_db = new_db1

    retriever = new_db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "content_of_eak_website",
        """
        This tool is designed for an LLM that interacts with 
        the content of the EAK website to retrieve documents. 
        The EAK acts as a compensation fund for various federal entities. 
        Its main responsibility is overseeing the implementation of 
        the 1st pillar (AHV/IV) and the family compensation fund. 
        The tool offers services related to:
            - Insurance
            - Contributions
            - Employer regulations
            - Pensions
        Furthermore, it provides insights into family allowances and 
        facilitates electronic data exchange with the EAK via connect.eak.
        """
    )
    tools = [tool]

    system_message = SystemMessage(
        content="""
        You are an expert for the eak_admin_website and:
        - Always answer questions citing the source.
        - The source is the URL you receive as a response from the eak_admin_website tool.
        - If you don't know an answer, state: "No source available, thus no answer possible".
        - Never invent URLs. Only use URLs from eak_admin_website.
        - Always respond in German.
        """
    )

    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     model=model,
                     temperature=0,
                     n=10,
                     verbose=True)

    agent_executor = create_conversational_retrieval_agent(
        llm, 
        tools, 
        verbose=False, 
        system_message=system_message,
        max_token_limit=3000) # heikel
 
    print(f"\nFrage: {query}")
    with get_openai_callback() as callback:
        answer = agent_executor({"input": query})
        print(f"\nAntwort: {answer['output']}\n\n")
        print(f"Total Tokens: {callback.total_tokens}")
        print(f"Prompt Tokens: {callback.prompt_tokens}")
        print(f"Completion Tokens: {callback.completion_tokens}")
        print(f"Total Cost (USD): ${callback.total_cost}")
    return answer


if __name__ == "__main__":

    QUESTIONS = [
        "Wann bezahlt die EAK jeweils die Rente aus?",
        "Was ist das SECO?",
        "Wer ist Kassenleiterin oder Kassenleiter der EAK?",
    ]

    for question in QUESTIONS:
        OPENAPI_API_KEY = "YOUR_API_KEY"
        SYS_PATH = "YOUR_SYSTEM_PATH"
        ask_agent(question, OPENAPI_API_KEY, SYS_PATH)
