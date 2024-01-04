
# 예제의 일부를 아래 블로그에서 퍼와서 사용함.
# https://bcho.tistory.com/1413
# LLM모델을 사용하기 위해서 필수적으로 사용되는 것이 프롬프트 인데 이를 엔지니어링 하는 것은 매우 중요하다.
# 원하는 답변을 얻을 수 있는 프롬프르틑 작성하고 재 사용할 수 있도록하고, 여러 프롬프르틑 구조화해서 
# 요청에 맞는 프롬프르틑 생성해야 한다. 이를 위핸 PromptTemplate를 간단히 소개한다.

### LangChain Chat Model Example  ### 
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate

# chat = ChatOpenAI(openai_api_key="{YOUR_API_KEY}")
llm = OpenAI()

template = PromptTemplate.from_template(
    "{city}에서 가장 {adjective} {topic}을 100자 안으로 알려줘."
)

prompt = template.format(adjective = "유명한", topic = "장소", city="서울")
print(prompt)
# print(llm(prompt))
print("\n")

prompt = template.format(adjective = "맛잇는", topic = "맛집", city="부산")
print(prompt)
# print(llm(prompt))
print("\n")

### 채팅 모델에서의 프롬프트 템플릿, system,human,AIMessage를 사용 해야 하기 때문에 포맷에 맞게 입력 해야함.
### 아래 코드에서 볼 수 있듯이 채팅 모델에서 템플릿을 사용 하면 작성하는 입장에서 조금 더 깔끔하다고 개인적으로 느껴짐.

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# chat = ChatOpenAI(openai_api_key="{YOUR_API_KEY}")
chat = ChatOpenAI()
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","너는 여행 가이드야"),
        ("human","나는 {Country}를 방문하려고 계획하고 있어."),
        ("ai","나는 {Country}의 여행 가이드야."),
        ("human","{user_input}")
    ]
)

prompt = chat_template.format_messages(Country="Korea",user_input="이 장소에서 가장 유명한 5가지 장소에 대해서 알려줘.")

print("Prompt :",prompt)
print("-"*30)

# aiMessage=chat.invoke(prompt)
# print(aiMessage)

## 프롬프트 조합 , 두개의 프롬프트를 단순히 연결
role_prompt = PromptTemplate.from_template("너는 {country}의 여행가이드 입니다.")
question_prompt = PromptTemplate.from_template("나에게 {country}에서 {interest} 알려 주세요.")

full_prompt=role_prompt + question_prompt
print(full_prompt.format(country="한국",interest="방문할 유명한 곳"))

## 프롬프트 파이프라이닝, 기본 프롬프트가 프롬프트 안의 변수를 지정하는 거라면 pipeline 프롬프트는 
## 각 기본 프롬프트를 어디에 삽입 할 것인가를 정하는 것 이라고 생각 한다.

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

full_template = """{role}

{question}

Please do not reply with anything other than information related to travel to {country} and reply “I cannot answer.
"""

full_prompt = PromptTemplate.from_template(full_template)
role_prompt = PromptTemplate.from_template("You are tour guide for {country}")
question_prompt = PromptTemplate.from_template("Please tell me about {interest} in {country}")

#composition
input_prompts = [
    ("role",role_prompt),
    ("question",question_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt,pipeline_prompts=input_prompts)

prompt_text = pipeline_prompt.format(
    country="Korea",
    interest="famous place to visit"
)

print(prompt_text)

## 만들어진 템플릿의 결과를 해석 하면, full_template에서 {role}과 {question}으로 role_prompt와 question_prompt가 삽입될 장소 지정.
## input_prompt에 full_prompt 프롬프트변수 이름인 role에 role_prompt를 지정하고, question에 question_prompt를 지정함
## pipelinePromptTemplate의 전체 프롬프트를 full_prompt로 저장한 후, pipeline_prompts에 포함될 프롬프트의 정보를 저장한
## input_prompt를 지정하면 prompt template가 생성된다. 생성 후엔 다른 template와 같이 format을 사용해서 문장 생성.