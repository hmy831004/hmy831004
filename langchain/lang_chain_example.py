from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
import os 

llm = ChatOpenAI(model="gpt-3.5-turbo-0613")

p1 = """주어진 텍스트에서 음식을 만들 수 있는 재료 이름을 추출해서 아웃풋을 한다.
<예>:
<텍스트>: ' 집에 왔는데 아우 정말 피곤한데 두부밖에 없고 야채는 감자밖에 없네.
나가기도 싫고. 뭐 해먹지.

<아웃풋>:
재료: 두부, 양파

<텍스트>:{text}"""
prompt1 = PromptTemplate(input_variables=["text"],template=p1)

p2= """ 주어진 재료로 만들 수 있는 쉬운 레시피를 찾아본다.
주어진 재료가 꼭 들어가야 하며, 구하기 힘든 재료는 제외한다.
전체 재료가 10가지를 넘지 않도록 한다.

예:
있는 재료: 식빵, 계란
음식: 프렌치 토스트
필요한 재료: 식빵, 계란, 소금, 설탕, 시나몬 혹은 꿀
레시피: 계란을 풀어 소금을 약간 치고 후라이팬에 굽는다.
설탕에 시나몬을 섞어 식빵위에 뿌려 먹거나 꿀을 뿌려서 먹는다.

{ingredients}
"""
prompt2 = PromptTemplate(input_variables=["ingredients"],template=p2)

chain1 = LLMChain(llm= llm, prompt = prompt1)
chain2 = LLMChain(llm= llm, prompt = prompt2)
recipe_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

# recipe = recipe_chain.run(input ="주말에 친구들 오는데 뭐 먹지? 소고기랑 감자랑 밀가루 있어.")
# print(recipe)


from openai import OpenAI
import ast
def get_recipes_with_ingredients(ingredients):
    return f"{ingredients} 를 다 볶아서 볶음밥으로 만들어 먹읍시다!"
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)

example_user_input = "주말에 친구들 오는데 뭐 먹지? 소고기랑 감자랑 밀가루 있어."
completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role":"user","content":example_user_input},
    ],
    functions=[
        {
            "name": "get_ingredients",
            "description":"주어진 텍스트에서 음식을 만들 수 있는 재료 이름을 뽑아낸다.",
            "parameters":{
                "type":"object",
                "properties":{
                    "ingredients":{
                        "type":"string",
                        "description":"음식재료",
                    },
                },
                "required":["ingredients"],
            },
        }
    ],
    function_call={"name":"get_ingredients"}
)
# print(completion.choices[0].message.content)
print(completion.choices[0].message)
message = completion.choices[0].message
if message.__dict__.get("function_call"):
    function_name = message.function_call.name
    args = ast.literal_eval(message.function_call.arguments)
    function_response = (
        get_recipes_with_ingredients(ingredients=args.get("ingredients")))
    completion_final = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role":"user","content":example_user_input},
            message,
            {
                "role":"function",
                "name": function_name,
                "content": str(function_response),
            }
        ]
    )
    
    import textwrap as tr
    my_str = completion_final.choices[0].message.content
    lines = tr.wrap(my_str,width=40)
    [print(x) for x in lines]