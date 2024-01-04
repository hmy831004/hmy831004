
# 예제의 일부를 아래 블로그에서 퍼와서 사용함.
# https://bcho.tistory.com/1409

# OPENAI API 키 관리
# -------------------
# 안전한 API 키 관리 방법:
# 1. API 키를 코드에 직접 삽입하는 것은 보안상 위험할 수 있습니다.
# 2. 대신, 로컬 환경에서 .env 파일을 생성하고 OPENAI_API_KEY="{YOUR_API_KEY}" 형식으로 키를 저장합니다.
# 3. 이렇게 하면 openai 패키지가 자동으로 .env 파일에서 API 키를 로드합니다.
#    이 기능은 openai 패키지 버전 1.0.0 이상에서 지원됩니다.

# 환경 변수 로딩:
# - 만약 openai 패키지가 자동으로 .env 파일에서 API 키를 로드하지 못하는 경우,
#   'dotenv' 패키지를 사용하여 수동으로 환경 변수를 로드할 수 있습니다.
# - 이를 위해 'from dotenv import load_dotenv'를 사용하고, 'load_dotenv()' 함수를 호출합니다.

# 예시:
# from dotenv import load_dotenv
# load_dotenv()  # .env 파일에서 환경 변수 로드


# OPENAPI KEY 관리 : openai 모델을 호출할 때 API key를 파라미터로 넘겨 주는 방식도 있지만 이는 외부에 노출이 될 위험이 있다.
# 로컬에 .env 파일을 작성하고 OPENAI_API_KEY="{YOUR_API_KEY}" 값을 저장해두면 
# OPENAI모델을 로드 할때 파일을 확인해서 값을 자동으로 입력 한다. * openai package version 1.0.0 이상 부터 지원 되는걸로 확인
# 만약 자동으로 key 값을 찾지 못할 경우 -> from dotenv import load_dotenv , load_dotenv() 를 실행해서 환경 파일을 로딩 한 후 실행


### lang chain을 이용해서 동기 방식으로 질문 하는 예제 ### 
from langchain import PromptTemplate
# from langchain.llms import OpenAI 
from langchain_community.llms import OpenAI



# llm = OpenAI(openai_api_key='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')
# OpenAI에서 호출 가능한 텍스트 생성 모델 : text-davinci-003, from langchain_community.llms import OpenAI
# OpenAI에서 호출 가능한 Chat 생성 모델 : gpt-3.5-turbo, gpt-4 , from langchain_community.chat_models import ChatOpenAI
llm = OpenAI(model = 'text-davinci-003')
prompt = "서울에서 유명한 거리 음식에 대해서 200자 이하로 설명해줘."
# print(llm.invoke(prompt)) # 동기 호출
# print(llm(prompt)) # call 호출 
# print(llm.generate([prompt])) # get meta information, 

# llm이 호출 가능한 LCEL(LangChain Expression Language)
# 동기 호출 : invoke, stream, batch, 
# 비동기 호출 : ainvoke, astream, abatch, astream_log , asyncio비동기 처리 모듈을 사용한 코딩 필요.


### Batch 호출 ###
prompts = [
    "What is top 5 Korean Street food?",
    "What is most famous place in Seoul?",
    "What is the popular K-Pop group?"
]
# print(f'{llm.batch(prompts)}')

### 동기 호출과 비동기 호출코드 및 시간 비교 ###
import asyncio
import time

prompt = "What is famous Korean food? Explain in 50 characters"

# Sync(동기) call
# start_time = time.perf_counter()
# for i in range(10):
#     result = llm.invoke(prompt)
#     print(result)
# end_time = time.perf_counter()
# print("Sync execution time:" ,(end_time-start_time))
# Async(비동기) call
async def invoke_async(llm):
    result = await llm.ainvoke(prompt)
    print(result)
async def invoke_parallel():
    tasks = [invoke_async(llm) for _ in range(10)]
    await asyncio.gather(*tasks)
start_time = time.perf_counter()
# asyncio.run(invoke_parallel())
end_time = time.perf_counter()
# print("Async execution time:" , (end_time-start_time))

# API 호출을 반복해서 많이 해야 하는 경우 비동기 처리를 해서 API요청을 하고 응답이 오기 전에 그다음 API요청을 진행하는 식으로 병렬적으로 처리가 가능하게함
# 동기 처리란 API 호출을 하고 나서 응답이 온 후 그다음 호출을 진행하는 방식을 말한다.


### 토큰 사용 및 요금 조회 ###
from langchain.callbacks import get_openai_callback

with get_openai_callback() as callback:
    prompt = "서울에서 유명한 거리 음식에 대해서 50자 이하로 설명해줘."
    print(llm.invoke(prompt))
    print(callback)




