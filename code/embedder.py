from sentence_transformers import SentenceTransformer, util
import os 
import glob 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import torch 
import gc
from fuzzywuzzy import fuzz 
import re 

DATA_ROOT = "/home/taco/Documents/projects/paper_analysis/data" 

historical_data = pd.read_excel(os.path.join(DATA_ROOT, "Historical_data.xlsx"))["오염문장"].values
room_df = pd.read_excel(os.path.join(DATA_ROOT, "Model_DB.xlsx"), sheet_name="격실 DB")
instrument_df = pd.read_excel(os.path.join(DATA_ROOT, "Model_DB.xlsx"), sheet_name="기기 DB")



def ko_sroberta(paper:pd.DataFrame, paper_name, score_threshold:float=0.65, precision:str=None) -> pd.DataFrame: 
    if precision: 
        embedder = SentenceTransformer("jhgan/ko-sroberta-multitask", model_kwargs={"torch_dtype":precision})
    
    else: 
        embedder = SentenceTransformer("jhgan/ko-sroberta-multitask") 
    
    sentences = paper["문장"] 
    paper_embedding = embedder.encode(sentences, convert_to_tensor=True) 
    query_embedding = embedder.encode(historical_data, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, paper_embedding)
    cos_scores = cos_scores.cpu()
    
    query_idx, paper_idx = np.where(cos_scores > score_threshold)
    score_values = cos_scores[query_idx, paper_idx] 
    res = pd.DataFrame({
        "Query": historical_data[query_idx],
        "오염문장": sentences[paper_idx],
        "score": score_values
    })
    
    paper_meta = paper[["문장", "페이지", "문장위치"]].rename(columns={"문장": "오염문장"})
    res = res.merge(paper_meta, on="오염문장", how="left")
    
    res = res.sort_values(by="score", ascending=False) 
    res = res.drop_duplicates(["오염문장"], keep="first")
    res = res.reset_index(drop=True)
    
    res.to_csv(os.path.join(res_root, "embedded", paper_name), index=False)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return res 


def to_lowercase(text:str): 
    if isinstance(text, str): 
        return re.sub(r'[A-Za-z]+', lambda x: x.group().lower(), text)
    else: 
        return text

def get_room_info(embedding_res:pd.DataFrame, similar_score_threshold:int) -> pd.DataFrame: 
    room_num = room_df["격실번호"]
    room_en = room_df["영문명"]
    room_ko = room_df["한글명"] # 이것만 비교할것 
    
    matched_room_nums = [] 
    matched_room_ens = [] 
    matched_room_kos = [] 
    
    for sentence in embedding_res["오염문장"]: 
        idx = find_best_matching_word(sentence, room_ko, similar_score_threshold) 
        if idx is not None: 
            matched_room_nums.append(room_num[idx]) 
            matched_room_ens.append(room_en[idx]) 
            matched_room_kos.append(room_ko[idx]) 
        else: 
            matched_room_nums.append(None) 
            matched_room_ens.append(None) 
            matched_room_kos.append(None) 
    
    embedding_res["격실번호"] = matched_room_nums 
    embedding_res["영문명"] = matched_room_ens 
    embedding_res["한글명"] = matched_room_kos 
    
    return embedding_res

def get_inst_info(embedding_res:pd.DataFrame, similar_score_threshold:int) -> pd.DataFrame: 
    inst_location = instrument_df["기능위치"]
    inst_name = instrument_df["기능위치명"] 
    inst_room = instrument_df["설치룸"] 
    int_name_lower_case = inst_name.apply(to_lowercase) # 이것만 비교할것
    contamin_sentence_eng = embedding_res["오염문장"].apply(to_lowercase)
    
    matched_locations = [] 
    matched_names = [] 
    mathced_room = [] 

    for sentence in contamin_sentence_eng: 
        idx = find_best_matching_word(sentence, int_name_lower_case, similar_score_threshold) 
        if idx is not None: 
            matched_locations.append(inst_location[idx]) 
            matched_names.append(inst_name[idx])
            mathced_room.append(inst_room[idx]) 
        else: 
            matched_locations.append(None) 
            matched_names.append(None) 
            mathced_room.append(None) 
            
    embedding_res["기능위치"] = matched_locations 
    embedding_res["기능위치명"] = matched_names 
    embedding_res["설치룸"] = mathced_room
    
    embedding_res["기능위치"] = embedding_res["기능위치"].astype("object")
    return embedding_res

def get_extra_info(embedding_res:pd.DataFrame, threshold): 
    inst_loc = instrument_df["기능위치"] 
    room_no = room_df["격실번호"]
    
    inst_loc_lower = inst_loc.apply(to_lowercase)
    room_no_lower = room_no.apply(to_lowercase) 
    for idx, sentence in enumerate(embedding_res["오염문장"].apply(to_lowercase)): 
        if pd.isna(embedding_res["기능위치"][idx]): 
            new_inst_loc_idx = find_best_matching_word(sentence, inst_loc_lower, threshold)
            if isinstance(new_inst_loc_idx, int): 
                embedding_res.loc[idx, "기능위치"] = inst_loc[new_inst_loc_idx] 
            else: 
                continue 
        
        if pd.isna(embedding_res["격실번호"][idx]): 
            new_room_no_idx = find_best_matching_word(sentence, room_no_lower, threshold) 
            if isinstance(new_room_no_idx, int): 
                embedding_res.loc[idx, "격실번호"] = room_no[new_room_no_idx] 
            else: 
                continue
    
    return embedding_res


def get_info(embedding_res, room_score=95, inst_score=95, extra_score=95):
    added_room_info = get_room_info(embedding_res, room_score) 
    added_inst_info = get_inst_info(added_room_info, inst_score) 
    final_res = get_extra_info(added_inst_info, extra_score)
    
    return final_res 

def find_best_matching_word(sentence, word_list, threshold): 
    best_idx = None 
    best_score = 0 
    for idx, word in enumerate(word_list): 
        if isinstance(word, str): 
            score = fuzz.partial_ratio(sentence, word) 
            if score >= threshold and score > best_score: 
                best_idx = idx 
                best_score = score 
        else: 
            continue
    
    return best_idx 


if __name__ == "__main__": 
    preprocessed_files_root = "/home/taco/Documents/projects/paper_analysis/data/preprocessed" 
    paper = list(glob.glob(f"{preprocessed_files_root}/*.csv"))
    res_root = "/home/taco/Documents/projects/paper_analysis/cosine90"
    
    if not os.path.exists(res_root): 
        os.makedirs(os.path.join(res_root, "embedded"))
        os.makedirs(os.path.join(res_root, "final"))
    
    for file in tqdm(paper): 
        file_name = os.path.basename(file)
        file = pd.read_csv(file)
        res = ko_sroberta(file, file_name, score_threshold=0.9) 
        final = get_info(res, 95, 95, 95) 
        final.to_csv(os.path.join(res_root, "final", file_name))