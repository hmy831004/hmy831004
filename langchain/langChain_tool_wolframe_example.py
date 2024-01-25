# pip install google-search-results
# pip install wolframealpha
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import load_tools
import os

os.environ['OPENAI_API_KEY'] = '{YOUR_OPENAI_API_KEY}'
os.environ['SERPAPI_API_KEY'] = '{YOUR_SERPAPI_API_KEY}'
os.environ['WOLFRAM_ALPHA_APPID'] = '{YOUR_FRAMALPHA_API_KEY}'

# llm = OpenAI(model ='gpt-3.5-turbo-instruct',temperature = 0)
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    HumanMessage(content="엄마의 나이는 34살이고 이모의 나이는 엄마보다 3살 더 적습니다. 고모의 나이는 이모보다 6살 더 많습니다.고모의 나이는 몇살입니까?")
]
llm = ChatOpenAI(model ='gpt-4-0613',temperature = 0)

tool_names = ['wolfram-alpha','serpapi']
tools = load_tools(tool_names)

agent = initialize_agent(tools,llm, agent='zero-shot-react-description',verbose=True)

# a = agent.run(messages)
b= agent(messages)

print(a)
print(llm(messages))