import os
import json
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
from tqdm import tqdm  # 导入进度条库

# default: Load the model on the available device(s)
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download

# 配置参数
class Args:
    model_path = "/data/weiyang/Qwen2.5-VL-72B-Instruct"
    conv_mode = "qwen_vl"
    answers_file = "./answers/answers.jsonl"
    temperature = 0.0
    top_p = 0.9
    num_beams = 1
    max_new_tokens = 1024  # 增加token长度确保输出完整
    seed = 42

def load_model(model_path):
    """加载支持多卡的本地模型"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto",
    attn_implementation="flash_attention_2", 
        ).eval()
    return processor, model

def process_question(line, processor):
    """处理多模态输入并添加bbox提示"""
    base_qs = line["text"]
    # 添加bbox格式要求提示
    bbox_prompt = "Please think step by step and output the bbox coordinates of the specified object in the format of bbox:[x1,x2,y1,y2]"
    qs = f"{base_qs}\n{bbox_prompt}"
    
    image_path = line["image"]
    [sub_folder, image_id] = image_path.split('/')
    file_path = hf_hub_download(
        repo_id="craigwu/vstar_bench",
        filename=image_id,
        subfolder=sub_folder,
        repo_type="dataset"
    )
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": file_path},
            {"type": "text", "text": qs}
        ]
    }]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs.to(model.device), text

def generate_response(model, inputs, processor):
    """多卡生成响应"""
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=Args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return output_text

if __name__ == "__main__":
    # 初始化多卡环境
    torch.manual_seed(Args.seed)
    
    print("Loading model...")
    processor, model = load_model(Args.model_path)
    
    dataset = load_dataset("craigwu/vstar_bench", split="test")
    
    os.makedirs(os.path.dirname(Args.answers_file), exist_ok=True)
    
    with open(Args.answers_file, "w") as f:
        # 添加进度条
        for idx, line in enumerate(tqdm(dataset, desc="Processing questions")):
            inputs, prompt = process_question(line, processor)
            response = generate_response(model, inputs, processor)
            response_text = response[0] if isinstance(response, list) else response
            # 提取bbox结果
            bbox = "未找到"  # 默认值

            start = response_text.find("[")
            end = response_text.find("]", start) if start != -1 else -1
            
            if start != -1 and end != -1:
                # 提取 `[x1,y1,x2,y2]` 部分
                bbox = response_text[start:end+1]  # +1 包含 `]`
            # print(bbox)
            # assert 0           
            result = {
                "question_id": idx,
                "prompt": prompt,
                "answer": response,
                "bbox": bbox,
                "gt_answer": line["label"],
                "model_id": "Qwen2.5-VL-7B-Instruct"
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
