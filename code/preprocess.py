from tqdm import tqdm 
import glob 
import re
from typing import List
import kss 
import os 
import pandas as pd 

def correct_spacing(page_data) -> List[str]:    
    res = re.sub(' +', ' ', page_data).strip()
    
    return res 


def seperate_sentence(page_data, page_idx):
    splitted_sentences = kss.split_sentences(page_data) 
    res = []
    for idx, sentence in enumerate(splitted_sentences):
        res.append({
            "문장": sentence,
            "페이지": page_idx+1, 
            "문장위치": idx+1
        })

    return res 


def remove_short_sentence(sentence_data, minimum_length): 
    res = [s for s in sentence_data if len(s["문장"]) > minimum_length]

    return res 


def main(paper_path:str, minimum_length_sentence=10): 
    paper_path = glob.escape(paper_path) 
    papers = list(glob.glob(f"{paper_path}/*.txt")) 
    
    preprocessed_sentences = [] 
    for paper in papers: 
        paper_no = os.path.basename(paper).split(".")[0] 
        paper_no = int(paper_no) 
            
        with open(paper, "r", encoding="utf-8") as f: 
            content = f.read()

        space_corrected = correct_spacing(content) 
        sentences = seperate_sentence(space_corrected, paper_no) 
        #short_sentence_removed = remove_short_sentence(sentences, minimum_length_sentence)

        for sentence_per_page in sentences: 
            preprocessed_sentences.append(sentence_per_page)
    
    return pd.DataFrame(preprocessed_sentences) 

if __name__ == "__main__": 
    txt_files_root = "/home/taco/Documents/projects/paper_analysis/data/txt" 
    txt_files = glob.glob(f"{txt_files_root}/*") 
    sv_path = "/home/taco/Documents/projects/paper_analysis/data/preprocessed"
    
    for file in tqdm(txt_files): 
        file_name = os.path.basename(file) 
        data = main(file) 
        data = data.sort_values(by=["페이지", "문장위치"])
        data.to_csv(os.path.join(sv_path, f"{file_name}.csv"), index=False)