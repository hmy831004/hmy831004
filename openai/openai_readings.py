from openai import OpenAI
from argparse import ArgumentParser
from time import time
import time as ti
import os, io
import re
import pandas as pd 
import base64
import requests

# python-dotenv 를 써서 키 숨기기 가능.  .env file안에 키를 삽입하고 라이브러리 호출하면 불러짐
# from dotenv import load_dotenv    ,  load_dotenv()
os.environ['OPENAI_API_KEY'] = '{YOUR_OPENAI_API_KEY}'
parser = ArgumentParser("u-con parser")
parser.add_argument("--type",help="select image type 1 or 2 or 3")
args = parser.parse_args()

class UconImageDesciription():

    def __init__(self) -> None: 
        self.client = OpenAI()

    def type1_chatgpt(self,input_text):
        """
            ① -1  본문(앞뒤) 텍스트 요약 : 본문(앞뒤) 텍스트 요약이 있는 이미지 유형
            ① -2   그림 정보의 캡션이 있는 경우  ==>  긁어지면.. 캡션요약하여 Alt text 처리
            ① -3 본문에서 그림 정보를 제공하는 경우 : 본문 그림정보 요약하여 Alt text 처리

            # openai chat api에서 사용 가능한 Parameter에 대한 설명.
            # temperature=, #1.0 이면 창의적인 생성, 0에 가까울 수록 일관 되고 예측 가능한 결과
            # frequency_penalty=, #1.0 에 가까울 수록 자주 나오는 단어에 패널티를 부과해서 일반적인 단어를 피하려고 노력함, 0 에가까울 수록 반복 단어가 많이 나옴 
            # presence_penalty=, # 이 설정은 새로운 개념이나 아이디어를 생성하는데 패널티부과, 1.0에 가까울수록 알려진 개념이나 아이디어를 피하려하고, 0에 가까울수록 알려진 아이디어를 더 사용하려함
            # function =, # 설정 하는 값에 따라서 원하는 결과를 추출 할 수 있게함. 추출된 Text에서 첫번째 나오는숫자. 대문에 영문등등
            # function_call 은 function과 한쌍으로 움직임, function에 지정된 name을 =  {name:'function에서 사용한 name'}
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a good assistant who can summarize and explain a sentence using the given text. Please keep your explanations to one sentence and no more than 20 characters. Also, please end your sentences with a noun."
                 },
                {"role": "user", "content": f"{input_text} Given the sentence, infer what the img* represents. Your response sentence should be a noun combination with no embellishments and an objective factual statement. Give me the result of your translation into Korean."}
            ]
        )
        
        text = response.choices[0].message.content
        text = re.sub('\s+',' ',text)
        return text

     
    def type3_chatcaption_local_image(self,img_path):
        """
            ③ -1 본문에 이미지 설명 없어서 real image captioning 필요
            이미지가 로컬에 존재하며 base64 encoding을 한 후 chatgpt에 입력으로 사용
        """
        base64_image = self.image_encoding(img_path)
        request_text = "이미지의 수식을 장애인들에게 알려주고 싶어. 이미지의 수식을 읽어줘."
        # request_text = "어떤 이미지일까?"
        ti.sleep(1)

        response =""
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )


        text = response.choices[0].message.content
        text = re.sub('\s+',' ',text)
        return text


    def type3_chatcaption(self,img_path):
        """
            ③ -1 본문에 이미지 설명 없어서 real image captioning 필요
            이미지 입력이 서버상에 존재하며 웹에서 검색 가능한 형태로 존재 할때 사용.
        """
        ti.sleep(1)
        request_text = "이미지의 수식을 장애인들에게 알려주고 싶어. 이미지의 수식을 읽어줘."
        # 꾸밈 문구가 없이 
        response =""
        try : 
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request_text},
                            {
                                
                                "type": "image_url",
                                "image_url": f"{img_path}",
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
        except Exception as e :
            print(e)
            # open ai response가 오지 않았을 때 10초후 다시 시도.
            if not response:
                print('not response 10 second sleep')
                ti.sleep(10)
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": request_text},
                                {
                                    
                                    "type": "image_url",
                                    "image_url": f"{img_path}",
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                
        text = response.choices[0].message.content
        text = re.sub('\s+',' ',text)
        return text

    def image_encoding(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
        
if __name__=="__main__":
    uid = UconImageDesciription()
    # result = uid.type3_chatcaption_local_image("../data/type_c/img_c.png")
    result = uid.type3_chatcaption_local_image("../data/reading/문자와식_1.png")
    print(result)