
# 예제의 일부를 아래 블로그에서 퍼와서 사용함.
# https://bcho.tistory.com/1411

# Langchain Cache을 이용해서 동일 혹은 유사한 프롬프트에 대한 결과를 캐싱하여 API호출을 줄일 수 있게 함.


### 내부 메모리를 이용한 캐싱, 어플리케이션 재시동시 캐시의 내용삭제 됨,  ### 
from langchain_community.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())
llm = OpenAI()
prompt = "서울에서 유명한 거리 음식에 대해서 200자 이하로 설명해줘."

with get_openai_callback() as callback:
    response = llm.invoke(prompt)
    print(response)
    print("Total Tokens",callback.total_tokens)

with get_openai_callback() as callback:
    llm.invoke(prompt)
    response = llm.invoke(prompt)
    print(response)
    print("Total Tokens",callback.total_tokens)
# 두번째 결과의 tokens는 0 으로 prompt와 똑같은 것이기 때문에 캐싱된 데이터를 출력함.

### 외부 캐싱, 외부 데이터베이스나 로컬에서 작동하는 SqlLite, Redis(메모리스토어), NoSQL 지원 ###
from langchain_community.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache
from redis import Redis
## 이 부분만 다르고 위의 예제와 동일 하기 때문에 중복 코드는 작성 하지 않는다.
set_llm_cache(RedisCache(redis_=Redis(host='{YOUR_REDIS_INTANCE_ADDRESS}',
  port={YOUR_REDIS_INSTANCE_PORT},
  password='{YOUR_REDIS_INSTANCE_PASSWORD}')))

### 시맨틱 캐싱 (Semantic Caching) ### 
# 위의 예제 까지는 prompt가 동일 해야지만 캐싱이 된다 하지만 문맥상 같은 prompt를 캐싱하고 싶을 때는
# 시맨틱 캐싱을 사용하여 prompt의 유사성을 검사한 후 캐싱에 사용할지 결정한다.


from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.cache import RedisCache
from redis import Redis

# Redis cache를 사용하고 추가로 embeddingAPI를 추가하면 된다.
RedisSemanticCache(redis_url="redis://default:{YOUR_REDIS_PASSWORD}@{YOUR_REDIS_INSTANCE_ADDRESS}:{YOUR_REDIS_INSTNACE_PORT}",
                    embedding=OpenAIEmbeddings())
)


llm = OpenAI(openai_api_key="{YOUR_API_KEY}")
prompt1 = "What is top 10 famous street foods in Seoul Korea in 200 characters"
prompt2 = "What is top 5 famous street foods in Seoul Korea in 200 characters"

with get_openai_callback() as callback:
    response = llm.invoke(prompt1)
    print(response)
    print("Total Tokens:",callback.total_tokens)

with get_openai_callback() as callback:
    llm.invoke(prompt)
    response = llm.invoke(prompt2)
    print(response)
    print("Total Tokens:",callback.total_tokens)

## 결과를 확인하면 두번째의 Total Tokens는 0 으로 유사한 prompt에 대해서도 캐싱이 수행 되었다.
## 하지만 첫 번째 프롬프트는 10개의 예제를 원했고 두번째는 5가지의 예제를 원했지만 첫번째 prompt에 의해서 10가지의 예제를 출력하기 때문에
## 무분별하게 캐싱을 사용하게 되면 원하지 않는 결과를 불러 올 수 있다.
## 세부적으로 컨트롤 하기 위해서는 유사도 분석 알고리즘을 지정하고 임베딩 알고리즘과 벡터 데이터베이스를 이용해 캐싱 시스템을 구축 해야한다.