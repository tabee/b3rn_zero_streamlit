''' 
https://python.langchain.com/docs/use_cases/
question_answering/how_to/conversational_retrieval_agents  '''
import langchain
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent, create_retriever_tool)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache


def ask_agent(query, openai_api_key, sys_path):
    '''Display the answer to a question.'''
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    new_db1 = FAISS.load_local(f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_4096',
                            embeddings)
    new_db2 = FAISS.load_local(f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_512',
                            embeddings)
    new_db3 = FAISS.load_local(f'{sys_path}/data/vectorstores/ch_ch_texts_faiss_index_4096',
                            embeddings)

    new_db1.merge_from(new_db2)
    new_db1.merge_from(new_db3)
    new_db = new_db1

    retriever = new_db.as_retriever()


    tool = create_retriever_tool(
        retriever,
        "eak_admin_website",
        """Searches and returns documents regarding the eak_admin_website.
        The eak_admin_website is the website of Eidg. Ausgleichskasse EAK."""""
    )
    tools = [tool]


    system_message = SystemMessage(
        content="""Du bist ein Experte der eak_admin_website
        und beanwortest Fragen immer unter Angabe der Quelle.
        Die Quelle ist die URL welche du im Tool eak_admin_website
        als Antwort erhälst. Wenn du eine Antwort nicht weisst,
        sage ich "keine Quelle, daher keine Antwort möglich".
        Du darfst URLs auf keinen Fall erfinden. Nur URLs aus
        eak_admin_website verwenden. Antworte immer auf deutsch.
        """
    )


    llm = ChatOpenAI(openai_api_key=openai_api_key,
                    model='gpt-3.5-turbo-16k',
                    temperature=0,
                    n=3,
                    verbose=True)

    agent_executor = create_conversational_retrieval_agent(
        llm, tools, verbose=False, system_message=system_message)

    print(f"\nFrage: {query}")
    with get_openai_callback() as callback:
        # geht: answer = agent_executor(query)
        answer = agent_executor({"input": query})
        print(f"Total Tokens: {callback.total_tokens}")
        print(f"Prompt Tokens: {callback.prompt_tokens}")
        print(f"Completion Tokens: {callback.completion_tokens}")
        print(f"Total Cost (USD): ${callback.total_cost}")
    print(f"\nFrage: {query}")
    print(f"\nAntwort: {answer['output']}")
    print("============================================================\n")
    return answer


if __name__ == "__main__":

    QUESTIONS = [
        "Wann bezahlt die EAK jeweils die Rente aus?",
        "Was ist das SECO?",
        "Wer ist Kassenleiterin oder Kassenleiter der EAK?",
    ]

    for question in QUESTIONS:
        ask_agent(question, openai_api_key=openai_api_key)
