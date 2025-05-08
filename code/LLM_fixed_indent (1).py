def Paper_Contamination(save_fullpath, folder_path, historical_data_fullpath, DB_fullpath, errorfile_fullpath):
    import pandas as pd
    import numpy as np
    from paper_ocr import paper_ocr
    import kss
    import re
    import os
    import warnings
    import time
    import json
    import glob
    import difflib
    from tqdm import tqdm
    from difflib import SequenceMatcher
    from llama_cpp import Llama

    # 점수 후처리 함수
    def extract_score(text):
        try:
            match = re.search(r"([0-9]+\.?[0-9]*)", text)
            if match:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
        except:
            return None
        return None

    # 경고 무시
    warnings.filterwarnings("ignore")
    start_time = time.time()

    # Sentence DB
    historical_data = pd.read_excel(
        historical_data_fullpath, engine="openpyxl")
    queries = historical_data.values.tolist()

    # 격실 및 기기 DB
    DB_room_df = pd.read_excel(DB_fullpath, sheet_name='격실 DB')
    DB_mc_df = pd.read_excel(DB_fullpath, sheet_name='기기 DB')

    # 폴더 내 파일 중 첫 두 개만 처리
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(
        os.path.join(folder_path, f))][:2]

    # Mistral 모델 로드 (쓰레드 수 1로 줄여 안정성 확보)
    mistral_model_path = "/home/taco/Documents/projects/jung/model/mis/mistral-7b-v0.1.Q4_K_M.gguf"
    llm = Llama(model_path=mistral_model_path, n_ctx=2048,
                n_threads=1, n_gpu_layers=30, verbose=False)

    result_line_list = []
    errorfile = []

    for paper in file_names:
        if paper == '.DS_Store':
            continue
        paper_path = os.path.join(folder_path, paper)

        try:
            data = paper_ocr(paper_path)
        except:
            errorfile.append(paper_path)
            continue

        newdata = [[re.sub(' +', ' ', sentence).strip()
                    for sentence in page] for page in data]

        sentence_temp = []
        if paper_path[-3:].lower() in ['txt', 'pdf']:
            for j_idx, page in enumerate(newdata):
                for item in page:
                    for sentence_idx, sentence in enumerate(kss.split_sentences(item)):
                        sentence_temp.append(
                            {'문장': sentence, '페이지': j_idx + 1, '문장위치': sentence_idx + 1})

        sentence_temp = [s for s in sentence_temp if len(s['문장']) > 10]
        sentence_df = pd.DataFrame(sentence_temp)
        sentence_list = [item['문장'] for item in sentence_temp]

        result_df_list = []
        similarity_cache = {}

        for query_row in queries:
            query = query_row[0]

            for target in sentence_list:
                key = (query, target)
                if key in similarity_cache:
                    score = similarity_cache[key]
                else:
                    if SequenceMatcher(None, query, target).ratio() < 0.2:
                        continue
                    try:
                        raw = llm(
                            f"""두 문장의 유사도를 0에서 1 사이 소수점 점수로만 숫자 하나만 출력하세요. 다른 말은 하지 마세요.\n문장1: {query}\n문장2: {target}\n답:""",
                            max_tokens=10,
                            stop=["\n"]
                        )
                        score = extract_score(raw['choices'][0]['text'])
                        similarity_cache[key] = score
                    except:
                        continue

                if score is not None and score > 0.3:
                    result_df_list.append(pd.DataFrame({
                        'Query': [query],
                        '오염문장': [target.strip()],
                        'score': [score]
                    }))

        if result_df_list:
            result = pd.concat(result_df_list, ignore_index=True)
            result = result.sort_values(by='score', ascending=False)
            result_fin = result.drop_duplicates(['오염문장'], keep='first')
            result_similarity = result_fin.reset_index(drop=True)
            result_similarity['페이지'] = None
            result_similarity['문장위치'] = None

            for index, row in result_similarity.iterrows():
                match_row = sentence_df[sentence_df['문장'] == row['오염문장']]
                if not match_row.empty:
                    result_similarity.at[index,
                                         '페이지'] = match_row['페이지'].values[0]
                    result_similarity.at[index,
                                         '문장위치'] = match_row['문장위치'].values[0]

            print(f"Paper: {paper}")
            print(result_similarity[['Query', '오염문장', 'score']].head())

    end_time = time.time()
    execution_time = end_time - start_time
    json_data = json.dumps({
        "TotalFiles": f"{len(file_names):,}",
        "ExecutionTimeInSeconds": f"{execution_time:,.0f}"
    })
    print(json_data)

    with open(errorfile_fullpath, 'w') as file:
        for item in errorfile:
            file.write(str(item) + '\n')
