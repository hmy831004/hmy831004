
# 예제의 일부를 블로그에서 퍼와서 사용함. 출처:https://bcho.tistory.com/1409

# OPENAI API 키 관리 에로 사항 및 해결 방법.
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

### OpenAI 텍스트 생성 모델과 챗 생성 모델 불러오기 ###
### lang chain을 이용해서 동기 방식으로 질문 하는 예제 ### 
# from langchain_community.llms import OpenAI

# from dotenv import load_dotenv
# load_dotenv()  # .env 파일에서 환경 변수 로드.

### OpenAI API 키 설정 및 모델 로드
# 사용 가능한 OpenAI 텍스트 생성 모델: text-davinci-003
# 사용 가능한 OpenAI 채팅 모델: gpt-3.5-turbo, gpt-4

# import os
# os.environ['OPENAI_API_KEY'] = '{YOUR_OPENAI_API_KEY}'
from langchain.llms import OpenAI
llm = OpenAI()
# llm = OpenAI(model = 'text-davinci-003')
prompt = "서울에서 유명한 거리 음식에 대해서 200자 이하로 설명해줘."
print(llm.invoke(prompt)) # 동기 호출: 직접적인 결과 반환
print(llm(prompt)) # call 호출: __call__ 메서드를 사용한 간편 호출
print(llm.generate([prompt])) # 메타 정보 포함: 추가 정보(예: 응답 시간, 토큰 수)를 얻을 수 있음


### Batch 호출 ###
prompts = [
    "한국 5대 길거리 음식은 무엇인가요?",
    "서울에서 가장 유명한 장소는 어디인가요?",
    "인기 있는 케이팝 그룹은 무엇인가요?"
]
print(f'{llm.batch(prompts)}')

### 동기 호출과 비동기 호출코드 및 시간 비교 ###
# llm이 호출 가능한 LCEL(LangChain Expression Language)
# 동기 호출 : invoke, stream, batch, 
# 비동기 호출 : ainvoke, astream, abatch, astream_log , asyncio비동기 처리 모듈을 사용한 코딩 필요.
import asyncio
import time

prompt = "유명한 한국 음식은 무엇인가요? 50자 이내로 설명하세요."


# 동기 호출 (Sync call)
# 동기 처리는 API 호출을 하고 응답을 받은 후에 다음 호출을 진행하는 방식임
start_time = time.perf_counter()
for i in range(10):
    result = llm.invoke(prompt)
    print(result)
end_time = time.perf_counter()
print("동기 호출 실행 시간:" ,(end_time-start_time))

# 비동기 호출 (Async call)
# 비동기 처리는 API 요청을 하고 응답을 기다리는 동안 다음 요청을 병렬적으로 진행하는 방식임
async def invoke_async(llm):
    result = await llm.ainvoke(prompt)
    print(result)
    
async def invoke_parallel():
    tasks = [invoke_async(llm) for _ in range(10)]
    await asyncio.gather(*tasks)

start_time = time.perf_counter()
asyncio.run(invoke_parallel())
end_time = time.perf_counter()
print("비동기 호출 실행 시간:" , (end_time-start_time))

# API 호출을 반복해서 많이 해야 하는 경우, 비동기 처리를 사용하면
# API 요청을 하고 응답이 오기 전에 다음 요청을 진행할 수 있어 효율적 일 수 있음.

### 토큰 사용 및 요금 조회 방법 ###
from langchain.callbacks import get_openai_callback

with get_openai_callback() as callback:
    prompt = "서울에서 유명한 거리 음식에 대해서 50자 이하로 설명해줘."
    print(llm.invoke(prompt))
    print(callback)




