"""
# 코드 주소 참고 
github 주소 : https://github.com/Beomi/KoAlpaca?tab=readme-ov-file
colab 주소 : https://colab.research.google.com/gist/Beomi/f163a6c04a869d18ee1a025b6d33e6d8/2023_05_26_bnb_4bit_koalpaca_v1_1a_on_polyglot_ko_12_8b.ipynb#scrollTo=FuXIFTFapAMI

# 필요 패키지, 로컬 환경에서 셋팅하기 위해서 적절한 버전을 선택해야함.
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git 
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q datasets
"""

"""
Issue : 
    peft는 torch version이 최소 1.13을 요구하기 때문에 torch 1.12를 가지고 있으면 사용 할 수 없다., CUDA-Version이 11.6 미만이라면 torch 1.13을 안정적으로 돌릴 수 없다.
    accelerator를 임포트 할 때,torch.distributed를 호출하고,거기서 /torch/fx/experimental/proxy_tensor.py의 get_innermost_proxy_mode 함수를 가져오는데, torch version이 2.0.0부터 생긴 함수이기 때문에 torch version이 2.0 미만이면 사용할 수 없다.
    model load할 때 cannot import name 'is_fx_tracing' from 'torch.fx._symbolic_trace' 메시지가 나오는데 이게 detectron라이브러리 버전이 올바르지 않아서 라고 한다.
    torch version이 (2.1) 미만 이면 torch.__init__ 에 _running_with_deploy 함수가 정의 되어 있지않음.
    torch version이 (2.1) 미만 이면 torch.distributed.distributed_c10d에 _find_pg_by_ranks_and_tag,_get_group_tag 함수가 정의 되어 있지않음.
    ... 
    위 작업들을 진행 하다가 torch version이 2.0 보다 낮으면,torch._C_. 함수들이 없거나, torch.fs 모듈들이 없는 경우가 많아서 하나씩 처리하기엔 무리가 있는 듯 보임.
    최대한 torch version >= 2.0까지는 맞춰야지 안정적으로 돌릴 수 있을 것으로 보인다.
"""


def print_trainable_parameters(model):
    trainable_params=0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )



from datasets import load_dataset
# koalpaca data load
data = load_dataset("beomi/KoAlpaca-v1.1a")
"""
data= 
DatasetDict({
    train: Dataset({
        features: ['instruction', 'output', 'url'],
        num_rows: 21155
    })
})
"""

# instruction settings : 학습 하고자 하는 포맷을 작성하고 그에 맞게 데이터를 만든다.
# data = data.map(
#     lambda x: 
#     {'text': f"### 명령어: {x['instruction']}\n\n###맥락: {x['input']}\n\n### 답변: {x['output']}<|endoftext|>" }
#     if x['input'] else 
#     {'text':f"### 명령어: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>"},
# )
data = data.map(
    lambda x: {'text': f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" }
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 12.8b 모델을 4bit로 로딩하면 GPU 메모리가 7191MiB 소모됨. 4bit로 로딩 하지 않을 시에는 : 50230MiB 
# polyglot 모델은 
model_id = "EleutherAI/polyglot-ko-12.8b"
# Bit Quantization options
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Model Quantization with INT4
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)


#gradient_checkpointing_enable : 역전파시 계산에 필요한 레이어 출력값을 activation메모리에 저장하는데 큰 모델이나, 메모리 제한이 있으면 실행이힘듬.
# 이를 일부 해소하기 위해서 중간 레이어의 일부만 메모리에 저장하고 나머지는 필요할 때 recompute하는 방식을 사용함, 메모리 사용량은 줄일 수 있지만 학습 시간이 오래걸림
model.gradient_checkpointing_enable()
from peft import prepare_model_for_kbit_training
"""
KBIT(Knowledge-Based Information Transfer, 특정한 태스크에 대한 사전 훈련된 지식을 모델에 효율적으로 전달하는 것을 의미함.)
- 모델의 파라미터를 적절하게 초기화,모델의 특정 부분을 조정하거나 특별한 층을 추가 할 수도 있음.
- 모델 구조 조정
    This method wraps the entire protocol for preparing a model before running a training. 
    This includes: 1- Cast the layernorm in fp32, 2- making output embedding layer require grads, 3- Add the upcasting of the lm head to fp32
"""
model = prepare_model_for_kbit_training(model)
# total_params = sum(p.numel() for p in model.parameters()) , # calurate model parameters
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        # warmup_steps=200,
        max_steps=500, ## 초소형만 학습: 10 step = 20개 샘플만 학습.
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
