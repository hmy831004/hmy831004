from openai import OpenAI
from argparse import ArgumentParser
from google.cloud import vision
from time import time
import time as ti
import os, io
import re
import pandas as pd 
import base64
import requests

# python-dotenv 를 써서 키 숨기기 가능.  .env file안에 키를 삽입하고 라이브러리 호출하면 불러짐
# from dotenv import load_dotenv    ,  load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '{YOUR_GOOGLE_API_KEY}' 
os.environ['OPENAI_API_KEY'] = '{YOUR_OPENAI_API_KEY}'
parser = ArgumentParser("u-con parser")
parser.add_argument("--type",help="select image type 1 or 2 or 3")
args = parser.parse_args()


class UconImageDesciription():

    def __init__(self) -> None: 
        self.client = OpenAI()
        # Instantiates a client
        self.vision_client = vision.ImageAnnotatorClient()
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
                # {"role": "system", "content": "You are a good assistant who can summarize and explain a sentence using a given text. You provide two types of explanations: short explanations(짧은 설명)  of 90 characters or less and long explanations(긴 설명) of 90 to 300 characters."},
                # {"role": "system", "content": "You are a good assistant who can summarize and explain a sentence using a given text. 짧은 설명과 긴 설명을 제공해줘. 짧은 설명은 꾸밈 문구가 없는 명사 조합 및 객관적 사실 중심, 긴 설명은 3문장 이하로 해줘. "},
                # {"role": "system", "content": "You are a good assistant who can summarize and explain a sentence using a given text. 짧은 설명과 긴 설명을 제공해줘. 짧은 설명은 설명은 30자 이하로 해주고, 긴 설명은 3문장 90자 이하로 해줘."},
                # {"role": "system", "content": "You are a good assistant who can summarize and explain a sentence using a given text. 짧은 설명과 긴 설명을 제공해줘. 짧은 설명은 설명은 30자 이하로 해주고 긴 설명은 3문장 이하 90자 이하로 해줘."},
                {"role": "system",
                 "content": "You are a good assistant who can summarize and explain a sentence using the given text. Please keep your explanations to one sentence and no more than 20 characters. Also, please end your sentences with a noun."
                 },
                # {"role": "user", "content": f"'{input_text}' 주어진 문장을 다음 조건에 맞게 요약 설명 해줘. 꾸밈 문구가 없는 명사 조합 및 객관적 사실 중심인 문장형 이고, 문장의 끝은 명사형 으로 종결 해줘."}
                # {"role": "user", "content": f"'{input_text}'  ."}
                # {"role": "user", "content": f"'{input_text}' 주어진 문장을 가지고 어떤 img인지 유추해줘. 꾸밈 문구가 없는 명사 조합 및 객관적 사실 중심인 문장형 으로 하고, 문장의 끝은 명사형 으로 종결 해줘."}
                {"role": "user", "content": f"{input_text} Given the sentence, infer what the img* represents. Your response sentence should be a noun combination with no embellishments and an objective factual statement. Give me the result of your translation into Korean."}
            ]
        )
        
        text = response.choices[0].message.content
        text = re.sub('\s+',' ',text)
        return text
    def type2_googleocr_chatgpt(self,img_file):
        """
            ② -1 이미지(그림 표,차트)OCR
            ②-2 그림 정보의 캡션이 있는 경우 ==> 안 긁어지면..  OCR 후 요약하여 Alt text 처리
        """
        
        # The name of the image file to annotate
        file_name = os.path.abspath(img_file)
        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        # For dense text, use document_text_detection
        # For less dense text, use text_detection
        response = self.vision_client.document_text_detection(image=image)
        ocr_text = response.full_text_annotation.text
        ocr_text = ocr_text.replace('\n',' ')

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                # {"role": "system", "content": "You are a good assistant who can summarize and explain a sentence using a given text. 설명문은 꾸밈 문구 없이 객관적 사실 중심 으로 하고 명사 조합으로 끝이 나게 만들어줘. 형용사와 부사는 사용 하지말아줘."},
                {"role": "system", "content": "You are a good assistant who can summarize and explain a sentence using a given text. 설명문은 꾸밈 문구 없이 객관적 사실 중심 으로 하고 명사 조합으로 끝이 나게 만들어줘. 형용사와 부사는 사용 하지말아줘."},
                {"role": "user", "content": f"'{ocr_text}' 주어진 Text들을 이용해 꾸밈 문구가 없이 조사와 동사를 활용한 명사 조합 및 객관적 사실 중심인 문장형 으로 설명해줘. 되도록 ,를 사용하지 않고 설명해줘. 한국어 15단어 이하로 알려줘."}
            ]
        )
        
        text = response.choices[0].message.content
        text = re.sub('\s+',' ',text)
        text = re.sub("[']",'',text)
        return text,ocr_text
     
    def type3_chatcaption_local_image(self,img_path):
        """
            ③ -1 본문에 이미지 설명 없어서 real image captioning 필요
            이미지가 로컬에 존재하며 base64 encoding을 한 후 chatgpt에 입력으로 사용
        """
        base64_image = self.image_encoding(img_path)
        # request_text = "어떤 이미지 인지 꾸밈 문구가 없이 조사와 동사를 활용한 명사 조합 및 객관적 사실 중심인 문장형 으로 설명해줘. 되도록 ,를 사용하지 않고 설명해줘. 문장의 끝은 명사형 으로 종결 해줘. 15단어 이하로 알려줘."
        request_text = "어떤 이미지 인지 설명 해줄 수 있을까?"
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
        #request_text ="어떤 이미지 인지 꾸밈 문구가 없는 명사 조합 및 객관적 사실 중심인 문장형 으로 설명해줘. 10단어 이하로 알려줘."},
        #request_text = "어떤 이미지 인지 명사 조합 및 객관적 사실 중심 으로 10단어 이하로 알려줘."
        # request_text = "어떤 이미지 인지를 조사와 동사를 활용한 명사 조합과 객관적 사실 중심인 문장형 으로 설명해줘. 되도록 ','를 사용하지 않고 15단어 이하로 설명해줘. 문장의 끝은 그림,이미지,사진,그래프 등의 단어로 표현해줘."
        # request_text = "어떤 이미지 인지 꾸밈 문구가 없이 조사와 동사를 활용한 명사 조합 및 객관적 사실 중심인 문장형 으로 설명해줘. 되도록 ','를 사용하지 않고 설명해줘. 문장의 끝은 명사형 으로 종결 해줘. 15단어 이하로 알려줘."
        request_text = "어떤 이미지 인지 조사와 동사를 활용한 명사 조합 및 객관적 사실 중심인 문장형 으로 설명해줘. 되도록 ','를 사용하지 않고 20단어 이하로 설명해줘. 문장의 끝은 그림,이미지,사진,그래프 등의 단어로 표현해줘."
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



uid = UconImageDesciription()
# How many time take = 1600.083sec
start = time()
df = pd.read_excel("data/Universal_contents_자동화_샘플_221207.xlsx",sheet_name="1유형(정제)",skiprows=6)
# img가 있는 row까지만 필터링.
df = df[~df.iloc[:,12].isna()]
# df = df.iloc[78:,:]
# df_read = pd.read_csv('ucon_type1.csv',sep='\t')
df = df.iloc[25:,:]
type1_list = [uid.type1_chatgpt(input_text=df.iat[i,12]) for i in range(df.shape[0])]
df_final = pd.DataFrame(type1_list,columns=["gpt"])
df_final['img'] = df.iloc[:,6]
df_final = df_final[['img','gpt']]
# 짧은 설명: .. 긴 설명: 의 gpt output을 나눠서 저장.
df_final[['gpt_short','gpt_long']] = df_final.gpt.str.split('.{1,2} 설명 *:',expand=True).iloc[:,[1,2]]
df_final = df_final.drop(columns=['gpt'])
df_final.to_csv('ucon_type1.csv',index=False,sep="\t")

end = time() 
print(f'How many time take = {end-start:.3f}sec')

# How many time take = 330.929sec
start = time()
data_path ='./data/type2' 
file_name = sorted(os.listdir(data_path))
full_file_name = [os.path.join(data_path,file) for file in file_name ]
type2_list = [uid.type2_googleocr_chatgpt(img_file=full_file_name[i]) for i in range(len(full_file_name))]
df_final = pd.DataFrame(type2_list,columns=["gpt","ocr_text"])
df_final['img'] = [os.path.splitext(os.path.basename(file))[0] for file in full_file_name]
df_final = df_final[['img','gpt','ocr_text']]
df_final.to_csv('ucon_type2.csv',index=False,sep="\t")
end = time() 
print(f'How many time take = {end-start:.3f}sec')

# 유형 3 서버에 이미지 있음. How many time take = 195.516sec 
start = time()
data_path ='./data/type3' 
file_name = sorted(os.listdir(data_path))
df = pd.read_excel("data/Universal_contents_자동화_샘플_20221214.xlsx",sheet_name="3유형",skiprows=6)
df = df.iloc[42:,:]
df = df[~df.iloc[:,8].isna()]
type3_list = [uid.type3_chatcaption(img_path=x) for x in ["https://i.ibb.co/tbHN3W9/getImage.jpg","https://i.ibb.co/T4Zr93D/20210507510935.jpg","https://i.ibb.co/H2PQQng/imageSrc.jpg"]]
type3_list = [uid.type3_chatcaption(img_path=df.iat[i,8]) for i in range(df.shape[0])]
df_final = pd.DataFrame(type3_list,columns=["gpt"])
df_final['img'] = df.iloc[:,7].reset_index(drop=True)
df_final = df_final[['img','gpt']]
df_final.to_csv('ucon_type3.csv',index=False,sep="\t")
end = time() 
print(f'How many time take = {end-start:.3f}sec')


### 유형 3 Local image exist
start = time()
data_path ='./data/type_c' 
file_name = sorted(os.listdir(data_path))
# file_name = [x for x in file_name if "02" in x ]
full_file_name = [os.path.join(data_path,file) for file in file_name ]
type3_local_list = [uid.type3_chatcaption_local_image(img_path=file_name) for file_name in full_file_name]
df_final = pd.DataFrame(type3_local_list,columns=["gpt"])
df_final['img'] = df.iloc[:,7]
df_final = df_final[['img','gpt']]
df_final.to_csv('ucon_type4.csv',index=False,sep="\t")
end = time() 
print(f'How many time take = {end-start:.3f}sec')

    

