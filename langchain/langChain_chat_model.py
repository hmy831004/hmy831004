
# 예제의 일부를 아래 블로그에서 퍼와서 사용함. 출처 : https://bcho.tistory.com/1412
# LLM 모델처럼 단일 출력을 지원하는 모델이 아닌 Chat모델은 기존의 대화 히스토리를 기반으로 질문에 대한 답변을 출력함.

### LangChain Chat Model Example  ### 
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

# chat = ChatOpenAI(openai_api_key="{YOUR_API_KEY}")
chat = ChatOpenAI()

# 대화 히스토리를 위한 메시지 타입
# SystemMessage: 개발자가 챗봇에게 내리는 명령 (역할, 가이드, 제약 사항 설정 등)
# HumanMessage: 사용자가 챗봇에게 하는 질문
# AIMessage: 챗봇의 답변

# 예시 대화 설정
messages = [
    SystemMessage(content="너는 여행 가이드 입니다. 사용자에게 여행 일정을 제공 해야해."),
    HumanMessage(content = "서울에서 가장 인기 있는 3가지 장소에 대해 알려줘.")
]

# 챗봇에게 대화 요청 및 응답 출력
aiMessage = chat.invoke(messages)
print(aiMessage)
print(chat.generate([messages]))

### 다음 대화를 이어서 호출 하고 싶을 때. , 
### 기존 질문과 AI 응답까지 합해서 다음 메시지를 부를때 넣어 주는식(하지만 이러면 토큰수가 커져서 요금 문제가 있음)
print("-"*30)
messages.append(aiMessage)
messages.append(HumanMessage(content='위 장소에 가기 위해서 어떤 이동수단을 이용 할 수 있니?'))
aiMessage2 = chat.invoke(messages)
print(aiMessage2)
# 주의: API 호출 시 토큰 수 제한 존재
# 대화가 길어질 경우, 토큰 수 제한을 넘지 않도록 관리 필요
# 필요한 경우, 이전 대화 내용을 요약하여 토큰 수 관리


### 위와 같이 messages에 직접 대화들을 추가 하는 방식도 있지만 LangChain에서 ChatMessageHistory 라는 클래스를 제공하는데 이를 사용 할 수도 있음.
from langchain.memory import ChatMessageHistory
messages = [
    SystemMessage(content="너는 여행 가이드 입니다. 사용자에게 여행 일정을 제공 해야해."),
    HumanMessage(content = "서울에서 가장 인기 있는 3가지 장소에 대해 알려줘.")
]
history = ChatMessageHistory()
history.add_user_message("서울에서 가장 인기 있는 3가지 장소에 대해 알려줘.")
aiMessage = chat.invoke(messages)
history.add_ai_message(aiMessage.content)
print(aiMessage.content)
print("-"*20)

history.add_user_message("위 장소에 가기 위해서 어떤 이동수단을 이용 할 수 있니?")
aiMessage2 = chat.invoke(history.messages)
history.add_ai_message(aiMessage2.content)
print(aiMessage2)
