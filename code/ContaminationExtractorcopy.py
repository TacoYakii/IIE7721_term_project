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
    import os
    import glob
    import pandas as pd
    import difflib
    from tqdm import tqdm

    from llama_cpp import Llama

    # 경고 무시
    warnings.filterwarnings("ignore")

    start_time = time.time()

    # Sentence DB
    historical_data = pd.read_excel(historical_data_fullpath, engine="openpyxl")
    queries = historical_data.values.tolist()

    # 격실 및 기기 DB
    DB_room_df = pd.read_excel(DB_fullpath, sheet_name='격실 DB')
    DB_mc_df = pd.read_excel(DB_fullpath, sheet_name='기기 DB')

    # 폴더 내의 파일 이름 가져오기
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)

    # Mistral 모델 로드
    mistral_model_path = "/home/taco/Documents/projects/jung/model/mis/mistral-7b-v0.1.Q4_K_M.gguf"
    llm = Llama(model_path=mistral_model_path, n_ctx=2048, n_threads=8, n_gpu_layers=30, verbose=False)

    result_line_list = []
    errorfile = []

    for paper in file_names:
        if paper == '.DS_Store':
            continue
        paper_name = folder_path + '/' + paper

        # OCR
        try:
            data = paper_ocr(paper_name)
        except:
            errorfile.append(paper_name)
            continue

        # 공백 제거
        newdata = []
        for page in data:
            preprocessed_page = [re.sub(' +', ' ', sentence).strip() for sentence in page]
            newdata.append(preprocessed_page)

        # 문장 분리
        sentence_temp = []
        if paper_name[-3:].lower() == 'txt' or paper_name[-3:].lower() == 'pdf':
            for j_idx, j in enumerate(newdata):
                sentences = []
                for item in j:
                    sentences.extend(kss.split_sentences(item))
                for sentence_idx, sentence in enumerate(sentences):
                    sentence_temp.append({'문장': sentence, '페이지': j_idx + 1, '문장위치': sentence_idx + 1})

        # 10자 이하 문장 제거
        min_length = 10
        sentence_temp = [s for s in sentence_temp if len(s['문장']) > min_length]
        sentence = pd.DataFrame(sentence_temp)
        sentence_list = [item['문장'] for item in sentence_temp]

        # 결과 비교
        result_df_list = []
        for query_row in queries:
            query = query_row[0]
            for idx, target in enumerate(sentence_list):
                prompt = f"""다음 두 문장은 얼마나 유사합니까? 0부터 1 사이의 점수로 정수나 소수로만 답해주세요.
문장1: {query}
문장2: {target}
점수:"""

                try:
                    output = llm(prompt, max_tokens=10, stop=["\n"])
                    score_text = output['choices'][0]['text'].strip()
                    score = float(score_text)

                    if score > 0.65:
                        result_df_list.append(pd.DataFrame({
                            'Query': [query],
                            '오염문장': [target.strip()],
                            'score': [score]
                        }))
                except:
                    continue

        if len(result_df_list) == 0:
            continue

        # 결과 DataFrame 생성
        result = pd.concat(result_df_list, ignore_index=True)
        result = result.sort_values(by='score', ascending=False)
        result_fin = result.drop_duplicates(['오염문장'], keep='first')
        result_similarity = result_fin.reset_index(drop=True)

        # 페이지, 문장위치
        result_similarity['페이지'] = None
        result_similarity['문장위치'] = None

        # 검색하여 일치하는 행의 '페이지'와 '문장위치' 추가
        for index, row in result_similarity.iterrows():
            query = row['오염문장']
            match_row = sentence[sentence['문장'] == query]
            if not match_row.empty:
                result_similarity.at[index, '페이지'] = match_row['페이지'].values[0]
                result_similarity.at[index, '문장위치'] = match_row['문장위치'].values[0]

        room_num = DB_room_df['격실번호'].tolist()
        room_en = DB_room_df['영문명'].tolist()
        room_ko = DB_room_df['한글명'].tolist()

        mc_num = DB_mc_df['기능위치'].tolist()
        mc_name = DB_mc_df['기능위치명'].tolist()
        mc_room= DB_mc_df['설치룸'].tolist()

        # 격실 한글명
        for word in room_ko:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(word)]
                if filtered_df.empty:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.9)).apply(pd.Series, dtype=object)
                    matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.
                else:
                    matching_rows = filtered_df

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    if sum(1 for item in room_ko if item == word) == 1:
                        indexes = room_ko.index(word)
                        room_num_rows = room_num[indexes]
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'],'페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': word, '기기명': "", '장소표현': room_num_rows, '기기표현': ""})
                    else:
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'],'페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': "", '기기표현': ""})

        # 격실 영문명
        for word in room_en:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(word)]
                if not filtered_df.empty:
                    matching_rows = filtered_df
                else:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                    if not similar_words.empty:
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.
                    else:
                        word_lower = word.lower()
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word_lower, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    if sum(1 for item in room_en if item == word) == 1:
                        indexes = room_en.index(word)
                        room_num_rows = room_num[indexes]
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'],'페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': room_num_rows, '기기표현': ""})
                    else:
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'],'페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': "", '기기표현': ""})

        #  격실번호
        for word in room_num:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(word)]
                if not filtered_df.empty:
                    matching_rows = filtered_df
                else:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                    if not similar_words.empty:
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.
                    else:
                        word = word.lower()
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    indexes = room_num.index(word)
                    room_en_rows = room_en[indexes]
                    if isinstance(room_en_rows, str):
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_en_rows, '기기명': "", '장소표현': word, '기기표현': ""})
                    else:
                        room_ko_rows = room_ko[indexes]
                        if isinstance(room_ko_rows, str):
                            result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_ko_rows, '기기명': "", '장소표현': word, '기기표현': ""})
                        else:
                            result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': "", '장소표현': word, '기기표현': ""})

        #  기기명
        for word in mc_name:
            if isinstance(word, str):
                # 기기명 한글 영어 구분
                if 'a' <= word <= 'z' or 'A' <= word <= 'Z':
                    filtered_df = result_similarity[result_similarity['오염문장'].str.contains(word)]
                    if not filtered_df.empty:
                        matching_rows = filtered_df
                    else:
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        if not similar_words.empty:
                            matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.
                        else:
                            word_lower = word.lower()
                            similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word_lower, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                            matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.

                    for i in range(len(matching_rows)):
                        result_rows = matching_rows.iloc[i]
                        indexes = mc_name.index(word)
                        mc_num_rows = mc_num[indexes]
                        mc_room_rows = mc_room[indexes]
                        if isinstance(mc_room_rows, str):
                            if mc_room_rows in room_num:
                                indexes = room_num.index(mc_room_rows)
                                room_en_rows = room_en[indexes]
                                if isinstance(room_en_rows, str):
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_en_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                else:
                                    room_ko_rows = room_ko[indexes]
                                    if isinstance(room_ko_rows, str):
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_ko_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                    else:
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': word, '장소표현': mc_room_rows, '기기표현': mc_num_rows})
                            else:
                                result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': word, '장소표현': "", '기기표현': mc_num_rows})
               
                else:
                    filtered_df = result_similarity[result_similarity['오염문장'].str.contains(word)]
                    if filtered_df.empty:
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.9)).apply(pd.Series, dtype=object)
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.
                    else:
                        matching_rows = filtered_df

                    for i in range(len(matching_rows)):
                        result_rows = matching_rows.iloc[i]
                        indexes = mc_name.index(word)
                        mc_num_rows = mc_num[indexes]
                        mc_room_rows = mc_room[indexes]
                        if isinstance(mc_room_rows, str):
                            if mc_room_rows in room_num:
                                indexes = room_num.index(mc_room_rows)
                                room_en_rows = room_en[indexes]
                                if isinstance(room_en_rows, str):
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_en_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                else:
                                    room_ko_rows = room_ko[indexes]
                                    if isinstance(room_ko_rows, str):
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_ko_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                    else:
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': word, '장소표현': mc_room_rows, '기기표현': mc_num_rows})
                            else:
                                result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': word, '장소표현': "", '기기표현': mc_num_rows})


        #  기기번호
        for word in mc_num:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(word)]
                if not filtered_df.empty:
                    matching_rows = filtered_df
                else:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                    if not similar_words.empty:
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.
                    else:
                        word = word.lower()
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        matching_rows = result_similarity[similar_words.notna().any(axis=1)] # 유사한 단어를 가진 행을 찾습니다.

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    indexes = mc_num.index(word)
                    mc_name_rows = mc_name[indexes]
                    mc_room_rows = mc_room[indexes]
                    if isinstance(mc_room_rows, str):
                        if mc_room_rows in room_num:
                            indexes = room_num.index(mc_room_rows)
                            room_en_rows = room_en[indexes]
                            if isinstance(room_en_rows, str):
                                result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_en_rows, '기기명': mc_name_rows, '장소표현': room_num_rows, '기기표현': word})
                            else:
                                room_ko_rows = room_ko[indexes]
                                if isinstance(room_ko_rows, str):
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': room_ko_rows, '기기명': mc_name_rows, '장소표현': room_num_rows, '기기표현': word})
                                else:
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': mc_name_rows, '장소표현': mc_room_rows, '기기표현': word})
                        else:
                            result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': mc_name_rows, '장소표현': "", '기기표현': word})
                    else:
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'],'문장위치': result_rows['문장위치'], 'score': result_rows['score'],'장소명': "", '기기명': mc_name_rows, '장소표현': "", '기기표현': word})

        # 그 외 문장들 추가
        present_sentence = [d['오염문장'] for d in result_line_list if '오염문장' in d]
        for i in range(len(result_similarity)):
            if result_similarity['오염문장'][i] not in present_sentence:
                result_line_list.append({'파일이름': paper_name, '오염문장': result_similarity['오염문장'][i], '페이지': result_similarity['페이지'][i],'문장위치': result_similarity['문장위치'][i],'score':result_similarity['score'][i],'장소명': "", '기기명': "", '장소표현': "", '기기표현': ""})
    
    result_line = pd.DataFrame(result_line_list)
    try:
        result_line = result_line.sort_values(by='score', ascending=False)
    except:
        pass
    result_line = result_line.drop_duplicates(['오염문장', '장소표현'], keep='first')
    try:
        result_line = result_line.drop(columns=['score'])
    except:
        pass
    result_line = result_line.reset_index(drop=True)
    
    if result_line.empty:
        result_line['파일이름'] = None
        result_line['오염문장'] = None
        result_line['페이지'] = None
        result_line['문장위치'] = None
        result_line['장소명'] = None
        result_line['기기명'] = None
        result_line['장소표현'] = None
        result_line['기기표현'] = None
    
    result_line.to_csv(save_fullpath, index=False, encoding='utf-8-sig')
    
    with open(errorfile_fullpath, 'w') as file:
        for item in errorfile:
            file.write(str(item) + '\n')
            
    # 코드 실행 종료 시간 기록
    end_time = time.time()
    
    # 실행 시간 계산
    execution_time = end_time - start_time    
    total_files = len(file_names)
    
    # JSON 형식으로 출력
    JSON_text = {
        "TotalFiles": f"{total_files:,}",
        "ExecutionTimeInSeconds": f"{execution_time:,.0f}"
    }
    
    json_data = json.dumps(JSON_text)
    print(json_data)