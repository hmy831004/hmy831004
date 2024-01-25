
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain,SequentialChain
from langchain.chains import LLMChain
from typing import List
import pandas as pd
import json

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '{YOUR_GOOGLE_API_KEY}' 
os.environ['OPENAI_API_KEY'] = '{YOUR_OPENAI_API_KEY}'

first_problems = [
    "파란색 장난감 자동차가 8대 있습니다. 오늘 어머니께서 빨간색 장난감 자동차 2대를 사 오셨습니다. 장난감 자동차는 모두 몇대 입니까?",
    "흰 강아지 1마리와 검은 강아지 9마리가 있습니다. 강아지는 모두 몇 마리입니까?",
    "어머니꼐서 사과 10개와 감 8개를 사 오셨습니다, 어머니꼐서 사오신 과일은 모두 몇 개 입니까?",
    "경선이의 책꽂이에는 동화책이 23권, 위인전이 46권 꽂혀 있습니다. 책꽂이에 꽂혀 있는 책은 모두 몇 권입니까?",
    "지원이는 동화책을 일주일 동안 96쪽 읽기로 했습니다. 오늘까지 80쪽을 읽었습니다. 몇 쪽을 더 읽어야 합니까?",
    "축구공은 41개, 야구공은 59개 있습니다. 야구공은 축구공보다 몇 개 더 많습니까?",
    "은진이가 아침에 일어나서 시계를 보았더니 7시 반이었습니다. 이때, 시계의 긴바늘은 숫자 'X'을 가리키고 있습니다.",
    "시계의 짧은바늘과 긴바늘이 겹치는 때는 'X'시 입니다.",
    "시계의 긴바늘이 1바퀴 돌면 짧은바늘은 숫자가 쓰여진 눈금을 몇칸 움직입니까?",
    "수연이는 사탕을 10개 가지고 있었습니다. 그중에서 2개는 친구에게 주고, 나머지는 동생과 똑같이 나누어 가졌습니다. 동생은 몇 개를 가졌습니까?",
    "7명이 타고 있던 버스에 첫째 정류장에서 1명이 더 타고 둘째 정류장에서 3명이 내렸습니다. 지금 버스 안에는 몇 명이 타고 있습니까?",
    "시계의 긴바늘은 숫자 6을 가리키고, 짧은바늘은 숫자 4와 5 사이를 가리키고 있습니다. 이 시계가 나타내는 시각을 말해 보시오.",
    "새롬이는 색종이를 76장 가지고 있었습니다. 오늘 미술 시간에 52장을 사용하고, 집에 올 때 15장을 새로 샀습니다. 새롬이가 지금 가지고 있는 색종이는 몇 장입니까?",
    "어떤 수에서 32를 빼야 할 것을 잘못하여 더하였더니 87이 되었습니다. 바르게 계산하면 얼마입니까?",
    # "시계의 긴바늘과 짧은바늘이 'ㄴ'자 모양이 되는 때는 몇 시입니까?",
]
second_problems=[
    "민우는 친구 한 명에게 사탕을 4개씩 나누어 주려고 합니다. 사탕 12개를 몇 명의 친구에게 나누어 줄 수 있습니까?",
    "성희는 우유를 하루에 몇 컵씩 7일 동안 마셨더니 모두 14컵을 마셨습니다. 하루에 몇 컵씩 마셨습니까?",
    "정민이는 63쪽짜리 위인전을 매일 같은 쪽수만큼 읽어서 9일 만에 모두 읽었습니다. 정민이는 하루에 위인전을 몇 쪽씩 읽었습니까?",
    "리본 한 개를 만드는 데 8cm의 끈이 필요합니다. 길이가 50cm인 끈으로 몇 개의 리본을 만들었더니 26cm가 남았습니다. 리본을 몇 개 만들었습니까?",
    "보라는 100쪽짜리 동화책을 매일 같은 쪽수만큼 8일동안 읽었더니 44쪽이 남았습니다. 보라는 하루에 동화책을 몇 쪽씩 읽었습니까?",
    "잠자리의 날개는 4장입니다. 잠자리 3마리의 날개는 모두 몇 장입니까?",
    "성냥개비 3개로 삼각형 한 개를 만들었습니다. 삼각형 7개를 만들려면 성냥개비가 모두 몇 개 필요합니까?",
    "도토리를 현희는 250개, 경호는 270개 주웠습니다. 경호는 현희보다 도토리를 몇 개 더 많이 주웠습니까?",
    "희경이는 258쪽 되는 동화책을 어제까지 216쪽을 읽었습니다. 동화책을 다 읽으려면 앞으로 몇 쪽을 더 읽어야 합니까?",
    "소희는 동화책 135쪽과 위인전 189쪽을 읽었습니다. 소희는 책을 모두 몇 쪽 읽었습니까?",
    "1년은 365일입니다. 오늘까지 256일이 지났다면, 며칠이 더 지나야 내년이 시작됩니까?",
    "서울역을 출발한 기차에는 223명이 타고 있었습니다. 다음 역에서 66명이 내리고 43명이 탔다면, 기차에는 몇 명이 타고 있겠습니까?",
    "한 상자에 옷이 7벌씩 들어 있습니다. 옷이 모두 21 벌입니다. 옷이 들어 있는 상자는 몇 상자입니까?",
    "친구 8명에게 초콜릿 32개를 똑같이 나누어 주려고 합니다. 한 사람에게 몇 개씩 나누어 줄 수 있습니까?",
    "남학생은 한 줄에 3명씩 6줄로 서 있고, 여학생은 한 줄에 5명씩 2줄로 서 있습니다. 이 학생들을 모두 모아 한 줄에 4명씩 세우려고 합니다. 몇 줄로 세울 수 있습니까?",
    "1번부터 9번까지 번호가 적혀 있는 놀이 기구가 3대있습니다. 수연이는 앞에서부터 24번째 번에 앉아 있습니다. 한 칸에 한 명씩 순서대로 놀이 기구를 탄다면, 수연이가 탈 놀이 기구는 몇 번입니까?",
    "'X'*5 = 20 일때 'X'의 값은? ",
    "첫 번째는 구슬1개 두 번째는 구슬 3개 세 번째는 5개 네 번째는 7개가 놓여 있습니다. 여섯째 번에는 몇개의 구슬이 놓일까요?",
]


### 기본 아이디어 - 한번에 하나씩 문제를 받아서 문제를 풀고 풀이를 받아 오는 것.
### 문제를 정확하게 풀 수 있는지 테스트 
def solve_math_problems(problems : List[str]):
    """
        문제의 리스트를 입력받고 문제를 푸는 함수,
    """
    llms = ChatOpenAI(model = 'gpt-4-0613')
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","너는 문제의 풀이를 알려주고 문제의 정답을 구하는 선생님이야. 문제 풀이를 보여준 다음 정답을 알려줘."),
            ("ai","알겠어 나에게 수학 문제를 주면 내가 정답을 구하고 풀이를 알려줄게."),
            ("human","{math_problem}"),
        ]
    )

    prompts = [chat_template.format_messages(math_problem=first) 
               for first in problems]
    answers=  [generation[0].text 
               for generation in llms.generate(prompts).generations]

    prompts_text = [prompt[-1].content for prompt in prompts]
    llm_output_json = [{'prompt':prompt,'answer':answer} for prompt,answer in zip(prompts_text,answers)]
    with open('math_ask_answer.json',"w",encoding="utf-8") as f:
        json.dump(llm_output_json,f,ensure_ascii=False,indent=4)

def create_math(content):
    """
        입력받은 content에 따라서 문제를 생성하는 함수.
    """
    llms = ChatOpenAI(model = 'gpt-4-0613')
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","너는 수학 문제를 만들어 주는 선생님이야. 초등학생의 문제를 만들거야. 문제의 포맷은 <문제번호>. <문제> 형식으로 만들어줘."),
            ("ai","알겠어 어떤 문제를 만들고 싶어 하는지 나에게 알려줘."),
            ("human","{content}")
        ]
    )
    prompt = chat_template.format_messages(content=content)
    text = llms.generate([prompt]).generations[0][-1].text
    return text


def create_math_return_json(content):
    """
        입력받은 content에 따라서 문제를 생성하는 함수.
    """
    # llms = ChatOpenAI(model = 'gpt-4-0613')
    llms = ChatOpenAI(model = 'gpt-4-1106-preview')
    
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","너는 수학 문제를 만들어 주는 선생님이야. 초등학생의 문제를 만들거야. "),
            ("ai","알겠어 어떤 문제를 만들고 싶어 하는지 나에게 알려줘."),
            ("human","{content} +\n\n 만든 결과를 JSON 형식 데이터로 출력하고,Key=<문제번호> Value=<문제> 형식으로 만들어주고 JSON이외에 다른 문자는 출력하지 마.")
        ]
    )
    prompt = chat_template.format_messages(content=content)
    text = llms.generate([prompt]).generations[0][-1].text
    return text

def create_math_with_fewshot(fewshot_examples):
    """
        예제를 입력하고 그 예제에 따라 문제를 생성하는 함수.
    """
    llms = ChatOpenAI(model = 'gpt-4-0613')
    problems = '\n'.join([str(i)+". "+example for i, example in enumerate(fewshot_examples,start=1)])
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","너는 수학 문제를 만들어 주는 선생님이야. 초등학생의 문제를 만들거야. 문제의 포맷은 <문제번호>. <문제> 형식으로 만들어줘."),
            ("ai","알겠어 어떤 문제를 만들고 싶어 하는지 나에게 알려줘. 예제를 알려주면 예제 문제의 성격과 비슷한 문제를 생성해줄게."),
            ("human","""아래에 만들고 싶은 문제 유형을 <문제번호>. <문제> 형식으로 제공할거야.\n
             {problems}
             \n 위 예제와 비슷한 문제를 10개 생성해줘.
             """)
        ]
    )
    prompt = chat_template.format_messages(problems=problems)
    text = llms.generate([prompt]).generations[0][-1].text
    return text

def create_math_and_resolve(content):
    """
        문제를 생성하고 -> 문제 해결 하는 함수.
        SequentialChain 을 통해 연속 진행       
    """
    llms = ChatOpenAI(model = 'gpt-4-0613')
    create_template = ChatPromptTemplate.from_messages(
        messages = 
        [
            ("system","너는 수학 문제를 만들어 주는 선생님이야. 초등학생의 문제를 만들거야. 문제의 포맷은 <문제번호>. <문제> 형식으로 만들어줘."),
            ("ai","알겠어 어떤 문제를 만들고 싶어 하는지 나에게 알려줘."),
            ("human","{content}")
        ]
    )
    create_chain = LLMChain(llm=llms,
                            prompt=create_template,
                            output_key="problems",
                            verbose=True ,)
    
    solve_template = ChatPromptTemplate.from_messages(
        [
            ("system","너는 문제의 풀이를 알려주고 문제의 정답을 구하는 선생님이야. <문제 풀이>\n<정답> 형식으로 알려줘."),
            ("ai","알겠어 나에게 수학 문제를 주면 내가 정답을 구하고 풀이를 알려줄게."),
            ("human","{problems}"),
        ]
    )
    solve_chain =  LLMChain(llm=llms,
                            prompt=solve_template,
                            output_key="solved_problems",
                            verbose=True)
    
    overall_chain = SequentialChain(
        chains=[create_chain,solve_chain],
        input_variables=['content'],
        output_variables = ['problems','solved_problems'],
        verbose=True
        )
    
    result = overall_chain(content)
    print(result)
    return result


if __name__=="__main__":
    
    solve_math_problems(first_problems+second_problems)
    print(create_math_return_json('초등학생에게 덧셈뺄셈에 대해서 알려주고 싶어. 문제를 쉽게 이해하기 위해서 서술형으로 풀어서 5문제 생성해줘.'))
    print(create_math('두자리 수의 곱하기와 나누기에 대해서 알려주고 싶어. 문제를 쉽게 이해하기 위해서 서술형으로 풀어서 5문제 생성해줘.'))
    print(create_math('초등학생에게 곱하기와 나누기 덧셈 뺄셈이 조합된 문제에 대해서 알려주고 싶어. 이 학생은 경시대회를 준비하고 있어, 경시대회에서 제출될만한 서술형 유형풀어서 5문제 생성해줘.'))
    print(create_math_with_fewshot(second_problems))
    create_math_and_resolve('초등학생에게 곱하기와 나누기에 대해서 알려주고 싶어. 문제를 쉽게 이해하기 위해서 서술형으로 풀어서 5문제 생성해줘.')