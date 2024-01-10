# OPENAI package가 1.0 이상 버전이 되면서 기존에 사용법과 달라져서, 한번 사용법을 총 정리하기 위한 파일임.

# open ai 셋팅
# import os
# os OPENAI_API_KEY 셋팅, 코드에 직접 넣기 때문에 별로 좋은 방법은 아님.
# os.environ["OPENAI_API_KEY"] = "{YOUR_OPENAO_API_KEY}"

# 로컬의 .env파일을 불러서 안에 OpenAIKey값을 불러오기 위한 함수
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# client = OpenAI(api_key='직접입력')
# api_key 직접입력 말고 두가지를 사용할 경우 그대로 선언해서 사용
client = OpenAI()
# 사용 가능한 모델 목록 확인
Models = client.models.list()
for m in Models.data:
    print(m.id)

# Text 생성 모델 사용법
# 2024-01-04 기준으로 text-davinci-003 모델이 사라졌고 대신해서, gpt-3.5-turbo-instruct Model을 사용하면 됨.
# OpenAI 문서 : https://platform.openai.com/docs/deprecations
output = client.completions.create(model='gpt-3.5-turbo-instruct',prompt='까마귀 울음소리')
print(output)

# chat 모델 사용법
output = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role':'user','content':'까마귀 울음소리는?'}
    ]
    )
print(output)


# # 랭체인을 이용한 LLM 일반 모델 사용법
from langchain.llms import OpenAI

# 랭체인 LLM 모델의 default모델도 text-davinci-003 모델인데 더이상 지원 하지 않기 때문에 model명을 입력해야함.
# llm = OpenAI()
llm = OpenAI(model ='gpt-3.5-turbo-instruct')
print(llm('까마귀울음소리는'))

# 랭체인을 이용한 LLM Chat 모델 사용법
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
llm = ChatOpenAI(model = 'gpt-3.5-turbo')
# LLM 호출
messages = [
    HumanMessage(content='고양이 울음 소리는?')
]
print(llm(messages))


# #LLm 동기식으로 호출 했을 때 걸리는 시간. ( 비동기에 비해 5배 이상 차이가남.)
import time

# 동기화 처리로 10번 호출하는 함수
def generate_serially():
    llm = OpenAI(model = 'gpt-3.5-turbo-instruct',temperature=0.9)
    for _ in range(10):
        resp = llm.generate(["안녕하세요!"])
        print(resp.generations[0][0].text)


# 시간 측정 시작
s = time.perf_counter()

# 동기화 처리로 10번 호출
generate_serially()

# 시간 측정 완료
elapsed = time.perf_counter() - s
print(f"{elapsed:0.2f} 초")

# LLM 비동기식 처리로 속도 업 ( 비동기란 한 작업이 끝나기 전에 작업을 시작 할 수 있는 것, 동기란 한 작업이 끝나야만 다음 작업이 가능함.)
import asyncio
# 이벤트 루프를 중첩하는 설정
import nest_asyncio

nest_asyncio.apply()

# 비동기 처리로 한 번만 호출하는 함수
async def async_generate(llm):
    resp = await llm.agenerate(["안녕하세요!"])
    print(resp.generations[0][0].text)

# 비동기 처리로 10회 호출하는 함수
async def generate_concurrently():
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0.9)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)

# 시간 측정 시작
s = time.perf_counter()

# 비동기 처리로 10회 호출
asyncio.run(generate_concurrently())

# 시간 측정 완료
elapsed = time.perf_counter() - s
print(f"{elapsed:0.2f} 초")


# ### 랭체인 템플릿 만들기.

from langchain.prompts import PromptTemplate

input_prompt = PromptTemplate(
    input_variables=["adjective","content"],
    template='{adjective} {content}이라고 하면?'
)

print(f'{input_prompt.format(adjective="멋진",content="동물")}')


# ### 답변 예시가 있는 템플릿 만들기
from langchain.prompts import FewShotPromptTemplate
examples = [
    {"input":"밝은","output":"어두운"},
    {"input":"재미있는","output":"지루한"}
]

example_prompt = PromptTemplate(
    input_variables=["input","output"],
    template = "입력: {input}\n출력: {output}",
)

prompt_from_string_example = FewShotPromptTemplate(
    input_variables=["input","output"],
    examples=examples,
    example_prompt= example_prompt,
    prefix=" 모든 입력에 대한  반의어를 입력하세요",
    suffix=" 입력: {adjective}\n출력:",
    example_separator="\n\n"
)

print(f'{prompt_from_string_example.format(adjective="큰")}')

#  Examples가 100개 존재할 때 모든 examples를 템플릿에 포함 시킬 수 없다. 
# 이럴 때 사용하기 위한, LengthBasedExampleSelector, SemanticSimilarityExampleSelector, MaxMarginalRelevanceSelector가 존재함.



## 재네릭 체인 , 인덱스 체인, 유틸리티 체인이 있음.
# LLMChain  사용자 입력을 기반으로 프롬프르틑 생성해 LLM 호출을 수행하는 chain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
template = """Q: {question}
A:"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

llm_chain = LLMChain(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0),
    prompt=prompt,
    verbose=True
)

question = "기타를 잘 치는 방법은?"
print(f'{llm_chain.predict(question=question)}')

# SimpleSequentialChain, 입출력이 하나씩 있는 여러 개의 체인을 연결 
from langchain.chains import SimpleSequentialChain

template = """
당신은 극작가입니다. 연극 제목이 주어졌을 때, 그 줄거리를 작성하는 것이 당신의 임무입니다.

제목:{title}
시놉시스:"""

prompt = PromptTemplate(
    input_variables=["title"],
    template=template,
)

chain1 = LLMChain(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0),
    prompt = prompt
)

template = """
당신은 연극 평론가입니다. 연극의 시놉시스가 주어지면 그 리뷰를 작성하는 것이 당신의 임무입니다.

시놉시스:{synopsis}
리뷰:"""

prompt = PromptTemplate(
    input_variables=["synopsis"],
    template=template
)

chain2 = LLMChain(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0),
    prompt = prompt
)

overall_chain = SimpleSequentialChain(
    chains=[chain1,chain2],
    verbose=True
)

print(f'{overall_chain.run("서울 랩소디")}')

# SequentialChain, 여러 개의 입출력을 가진 체인을 연결하는 체인
from langchain.chains import SequentialChain
template = """
당신은 극작가입니다. 연극 제목이 주어졌을 때, 그 줄거리를 작성하는 것이 당신의 임무입니다.

제목:{title}
시대:{era}
시놉시스:"""

prompt = PromptTemplate(
    input_variables=["title","era"],
    template=template
)

chain1 = LLMChain(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0),
    prompt = prompt,
    output_key="synopsis"
)
template = """
당신은 연극 평론가입니다. 연극의 시놉시스가 주어지면 그 리뷰를 작성하는 것이 당신의 임무입니다.

시놉시스:{synopsis}
리뷰:"""

prompt = PromptTemplate(
    input_variables=["synopsis"],
    template=template
)

chain2 = LLMChain(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0),
    prompt = prompt,
    output_key="review"
)



overall_chain = SequentialChain(
    chains=[chain1,chain2],
    input_variables=["title","era"],
    output_variables=["synopsis","review"],
    verbose=True
)

print(f"{overall_chain({'title':'서울 랩소디','era':'100년 후의 미래'})}")

## 인덱스 체인- 공개되지 않은 개인의 고유 데이터를 이용해 질의응답을 하기 위한 체인임. 인덱스 조작을 수행하며 라마인덱스와 유사한 기능을 함.
## RetrievalQA, RetrievalQAWithSourcesChain, SummarizaeChain 3가지 인덱스 체인이 있음.

from langchain.text_splitter import CharacterTextSplitter

with open("../data/langchain/akazukin_all.txt") as f:
    test_all = f.read()
    
# 청크 분할, 데이터셋을 원하는크기로 나눠주는 함수, 문자를 자를 때 내용을 이어지게 하기 위해 overlap을 일정수준 잡아야함.
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size=300,
    chunk_overlap=20
)

texts = text_splitter.split_text(test_all)
print(texts)
# print(len(texts))
# for text in texts:
#     print(text[:10], ":", len(text))

# 벡터 데이터베이스 생성
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# 벡터 데이터베이스 생성
docsearch = FAISS.from_texts(
    texts =texts, # 청크 배열
    embedding= OpenAIEmbeddings()  
)

# 질의응답 체인 만들기
from langchain.chains import RetrievalQA
"""
cahin_type
- stuff는 관련 데이터를 컨텍스트로 프롬프트에 담아 언어 모델에 전달하는 방식,
장점은 LLM을 한번만 호출하면 되고, 텍스트 생성시 LLM이 한번에 모든 데이터에 접근할 수 있다. 
단점은 LLM에는 컨텍스트 길이 제한이 있어 큰 데이터에는 작동하지 않는다는 점

- map_reduce는  관련 데이터를 청크로 분할하고, 청크별로 프롬프트를 생성해서 LLM을 호출하고 마지막으로 모든 결과를 결합하는 프롬프트로 LLM을 호출하는 방식
장점은 stuff보다 더 큰 데이터에도 작동해서 청크 단위의 LLM 호출을 병렬로 실행할 수 있다는 점.
단점은 stuff보다 더 많은 LLM호출이 필요하고 마지막 결합에서 일부 정보가 손실될 수 있다는 점.

- refine은 관련 데이터를 청크로 나누고 첫번째 청크마다 프롬프트를 생성해 LLM호출하고 그 출력과 함께 다음 청크에서 프롬프트를 생성해서 LLM을 호출하고 이를 반복
장점은 좀 더 관련성 높은 컨텍스트를 가져올 수 있다는 점과 map_reduce보다 손실이 적을 수 있다는 점
단점은 stuff보다 더 많은 호출이 필요하고, 청크 LLM 호출을 병렬로 실행할 수 없다는 점. 

- map_rerank는 관련 데이터를 청크로 나누고, 청크마다 프롬프트를 생성해서 LLM을 호출하고 그 답변이 얼마나 확실한지를 나타내는 점수를 표시하고 이 점수에 따라 응답의 순위를 매겨 높은 점수를 응답받은 응답을 반환하는 방식
장점은 map_reduce와 유사하며, map_reduce보다 호출 횟수가 적다는 점, 하나의 문서에서 하나의 간단한 답변이 있을 때 가장 적합한 방식.
단점은 문서 간 정보를 결합할 수 없다는 점.


리트리버(retriever)는 컴퓨터 용어로는 특정 키워드나 특정 조건에 따라 데이터베이스나 웹상의 정보를 수집해서 사용자에게 제공하는 정보 검색 시스템을 나타내는 용어.
랭체인에서는 다양한 검색 방식으로 전환하기 위해 문서 검색 방식을 추상화한 모듈을 리트리버라고 부르고 있음.
"""
qa_chain = RetrievalQA.from_chain_type(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0),
    chain_type = "stuff", #체인 종류
    retriever=docsearch.as_retriever(), #리트리버
)

print(f'{qa_chain.run("미코의 소꿉친구 이름은?")}')

## RetrievalQAWithSourcesChain은 소스가 있는 질의응답을 하기 위한 체인, 소스는 질의응답을 할 때 사용한 정보 출처(웹페이지라면 URL, 책이라면 책의 몇페이지 인지 등의 정보를 나타냄)
# 메타데이터를 준비하고 벡터 데이터베이스 생성시에 삽입해준다.

## Summarize 체인
from langchain.docstore.document import Document
# 청크 배열을 문서 배열로 변환
docs = [Document(page_content=t) for t in texts]

from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature = 0),
    chain_type = "map_reduce",
    
)
print(f'{chain.run(docs)}')




