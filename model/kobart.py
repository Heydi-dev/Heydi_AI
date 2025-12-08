# model/kobart.py
# 코드 설명: KoBART 모델을 사용하여 일기 내용으로부터 주제 태그를 생성하는 모델 학습 스크립트입니다.
# 사용된 모델명: gogamza/kobart-base-v1
# 사용된 데이터셋: diary_tag_dataset.json
# 학습된 모델은 ./model/final_diary_tag_model 경로에 저장됩니다.
import json
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ==========================================
# [경로 설정] 현재 스크립트 위치 기준
# ==========================================
# 현재 파일(kobart.py)이 있는 폴더 경로 (예: .../Project/model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 경로 (model 폴더의 상위 폴더에 있다고 가정: ../diary_tag_dataset.json)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diary_tag_dataset.json")

# 결과물 저장 경로 (모두 model 폴더 내부로 설정)
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "final_diary_tag_model")

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
def load_data(file_path):
    print(f">>> 데이터셋 로드 경로: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['target_text'] = df['tags'].apply(lambda x: ', '.join(x))
    return df

# 데이터 로드
if os.path.exists(DATASET_PATH):
    df = load_data(DATASET_PATH)
else:
    # 파일이 없을 경우 테스트용 더미 데이터 사용 (에러 방지)
    print("⚠️ 데이터셋 파일을 찾을 수 없어 샘플 데이터로 대체합니다.")
    sample_data = [
        {"content": "테스트 일기입니다.", "tags": ["테스트", "샘플"]}
    ]
    df = pd.DataFrame(sample_data)
    df['target_text'] = df['tags'].apply(lambda x: ', '.join(x))

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# ---------------------------------------------------------
# 2. 토크나이저 및 모델 불러오기
# ---------------------------------------------------------
model_name = "gogamza/kobart-base-v1"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# ---------------------------------------------------------
# 3. 데이터셋 클래스 정의
# ---------------------------------------------------------
class DiaryTagDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=128, max_target_len=32):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = row['content']
        target_text = row['target_text']

        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_input_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            text_target=target_text, 
            max_length=self.max_target_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        labels_ids = labels["input_ids"].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels_ids
        }

train_dataset = DiaryTagDataset(train_df, tokenizer)
val_dataset = DiaryTagDataset(val_df, tokenizer)

# ---------------------------------------------------------
# 4. 학습 설정
# ---------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,            # ./model/results
    logging_dir=LOG_DIR,              # ./model/logs
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=10,
    eval_strategy="epoch",            # evaluation_strategy 대체
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ---------------------------------------------------------
# 5. 트레이너 및 학습
# ---------------------------------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,       # tokenizer 대체 (FutureWarning 해결)
)

print(f">>> 학습 시작... 결과는 {OUTPUT_DIR}에 저장됩니다.")
trainer.train()

# ---------------------------------------------------------
# 6. 모델 저장
# ---------------------------------------------------------
print(f">>> 모델 저장 중: {MODEL_SAVE_DIR}")
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(">>> 모든 작업 완료.")