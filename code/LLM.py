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
    import concurrent.futures
    from tqdm import tqdm
    from difflib import SequenceMatcher
    from multiprocessing import Pool, cpu_count
    from llama_cpp import Llama
    # 점수 후처리 함수

    def extract_score(text):
        try:
            match = re.search(r"([0-9]+\.?[0-9]*)", text)
            if match:
                return float(match.group(1))  # 범위 제한 없이 다 받음
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

    # 폴더 내의 파일 이름 가져오기
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)

    # Mistral 모델 로드
    mistral_model_path = "/home/taco/Documents/projects/jung/model/mis/mistral-7b-v0.1.Q4_K_M.gguf"
    llm = Llama(model_path=mistral_model_path, n_ctx=512,
                n_threads=1, n_gpu_layers=30, verbose=False)

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
            preprocessed_page = [
                re.sub(' +', ' ', sentence).strip() for sentence in page]
            newdata.append(preprocessed_page)

        # 문장 분리
        sentence_temp = []
        if paper_name[-3:].lower() == 'txt' or paper_name[-3:].lower() == 'pdf':
            for j_idx, j in enumerate(newdata):
                sentences = []
                for item in j:
                    sentences.extend(kss.split_sentences(item))
                for sentence_idx, sentence in enumerate(sentences):
                    sentence_temp.append(
                        {'문장': sentence, '페이지': j_idx + 1, '문장위치': sentence_idx + 1})

        # 10자 이하 문장 제거
        min_length = 10
        sentence_temp = [s for s in sentence_temp if len(s['문장']) > min_length]
        sentence = pd.DataFrame(sentence_temp)
        sentence_list = [item['문장'] for item in sentence_temp]

        # ------------------- 결과 비교 -------------------
        # 오염문장(query)와 OCR 문장(target)을 Mistral 모델로 비교
        # SequenceMatcher로 1차 필터링 (0.3 미만 제외)
        # LLM의 출력에서 유사도 점수(0~1)만 추출
        # 점수가 0.65 초과인 경우에만 결과에 포함

        result_df_list = []
        similarity_cache = {}  # 문장쌍 유사도 캐시
        all_scores = []  # (query, target, score) 저장용 – 나중에 정규화에 사용

        for query_row in queries:
            query = query_row[0].strip()

            for target in sentence_list:
                target = target.strip()

                key = (query, target)
                if key in similarity_cache:
                    score = similarity_cache[key]
                else:
                    # SequenceMatcher로 1차 필터링
                    if SequenceMatcher(None, query, target).ratio() < 0.1:
                        continue

                    try:
                        prompt = f"""두 문장의 유사도를 0에서 1 사이 소수점 점수로만 숫자 하나만 출력하세요. 다른 말은 하지 마세요.\n문장1: {query}\n문장2: {target}\n답:"""
                        output = llm(prompt, max_tokens=10, stop=["\n"])
                        score = extract_score(output['choices'][0]['text'])
                        similarity_cache[key] = score
                    except:
                        continue

                if score is not None:
                    all_scores.append((query, target, score))  # 정규화용으로 저장만 해둠

        if all_scores:
            score_values = [s[2] for s in all_scores]
            min_score = min(score_values)
            max_score = max(score_values)

            def normalize(score):
                if max_score == min_score:
                    return 1.0
                return (score - min_score) / (max_score - min_score)

    # 필터링 및 결과 저장
            for query, target, score in all_scores:
                norm_score = normalize(score)
                if norm_score > 0.65:
                    result_df_list.append({
                        'Query': query,
                        '오염문장': target,
                        'score': score,
                        'normalized_score': norm_score
                    })
        # 결과 DataFrame 생성
        result = pd.DataFrame(result_df_list)  # 리스트[dict] → DataFrame
        result = result.sort_values(by='normalized_score', ascending=False)
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
                result_similarity.at[index,
                                     '문장위치'] = match_row['문장위치'].values[0]

        room_num = DB_room_df['격실번호'].tolist()
        room_en = DB_room_df['영문명'].tolist()
        room_ko = DB_room_df['한글명'].tolist()

        mc_num = DB_mc_df['기능위치'].tolist()
        mc_name = DB_mc_df['기능위치명'].tolist()
        mc_room = DB_mc_df['설치룸'].tolist()

        # 격실 한글명
        for word in room_ko:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(
                    word)]
                if filtered_df.empty:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                        word, x.split(), n=1, cutoff=0.9)).apply(pd.Series, dtype=object)
                    # 유사한 단어를 가진 행을 찾습니다.
                    matching_rows = result_similarity[similar_words.notna().any(
                        axis=1)]
                else:
                    matching_rows = filtered_df

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    if sum(1 for item in room_ko if item == word) == 1:
                        indexes = room_ko.index(word)
                        room_num_rows = room_num[indexes]
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                '문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': room_num_rows, '기기표현': ""})
                    else:
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                '문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': "", '기기표현': ""})

        # 격실 영문명
        for word in room_en:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(
                    word)]
                if not filtered_df.empty:
                    matching_rows = filtered_df
                else:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                        word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                    if not similar_words.empty:
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]
                    else:
                        word_lower = word.lower()
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                            word_lower, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    if sum(1 for item in room_en if item == word) == 1:
                        indexes = room_en.index(word)
                        room_num_rows = room_num[indexes]
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                '문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': room_num_rows, '기기표현': ""})
                    else:
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                '문장위치'], 'score': result_rows['score'], '장소명': word, '기기명': "", '장소표현': "", '기기표현': ""})

        #  격실번호
        for word in room_num:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(
                    word)]
                if not filtered_df.empty:
                    matching_rows = filtered_df
                else:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                        word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                    if not similar_words.empty:
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]
                    else:
                        word = word.lower()
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                            word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]

                for i in range(len(matching_rows)):
                    result_rows = matching_rows.iloc[i]
                    indexes = room_num.index(word)
                    room_en_rows = room_en[indexes]
                    if isinstance(room_en_rows, str):
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                '문장위치'], 'score': result_rows['score'], '장소명': room_en_rows, '기기명': "", '장소표현': word, '기기표현': ""})
                    else:
                        room_ko_rows = room_ko[indexes]
                        if isinstance(room_ko_rows, str):
                            result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                    '문장위치'], 'score': result_rows['score'], '장소명': room_ko_rows, '기기명': "", '장소표현': word, '기기표현': ""})
                        else:
                            result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                    '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': "", '장소표현': word, '기기표현': ""})

        #  기기명
        for word in mc_name:
            if isinstance(word, str):
                # 기기명 한글 영어 구분
                if 'a' <= word <= 'z' or 'A' <= word <= 'Z':
                    filtered_df = result_similarity[result_similarity['오염문장'].str.contains(
                        word)]
                    if not filtered_df.empty:
                        matching_rows = filtered_df
                    else:
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                            word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        if not similar_words.empty:
                            # 유사한 단어를 가진 행을 찾습니다.
                            matching_rows = result_similarity[similar_words.notna().any(
                                axis=1)]
                        else:
                            word_lower = word.lower()
                            similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                                word_lower, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                            # 유사한 단어를 가진 행을 찾습니다.
                            matching_rows = result_similarity[similar_words.notna().any(
                                axis=1)]

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
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                            '문장위치'], 'score': result_rows['score'], '장소명': room_en_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                else:
                                    room_ko_rows = room_ko[indexes]
                                    if isinstance(room_ko_rows, str):
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                                '문장위치'], 'score': result_rows['score'], '장소명': room_ko_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                    else:
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                                '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': word, '장소표현': mc_room_rows, '기기표현': mc_num_rows})
                            else:
                                result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                        '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': word, '장소표현': "", '기기표현': mc_num_rows})

                else:
                    filtered_df = result_similarity[result_similarity['오염문장'].str.contains(
                        word)]
                    if filtered_df.empty:
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                            word, x.split(), n=1, cutoff=0.9)).apply(pd.Series, dtype=object)
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]
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
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                            '문장위치'], 'score': result_rows['score'], '장소명': room_en_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                else:
                                    room_ko_rows = room_ko[indexes]
                                    if isinstance(room_ko_rows, str):
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                                '문장위치'], 'score': result_rows['score'], '장소명': room_ko_rows, '기기명': word, '장소표현': room_num_rows, '기기표현': mc_num_rows})
                                    else:
                                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                                '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': word, '장소표현': mc_room_rows, '기기표현': mc_num_rows})
                            else:
                                result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                        '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': word, '장소표현': "", '기기표현': mc_num_rows})

        #  기기번호
        for word in mc_num:
            if isinstance(word, str):
                filtered_df = result_similarity[result_similarity['오염문장'].str.contains(
                    word)]
                if not filtered_df.empty:
                    matching_rows = filtered_df
                else:
                    similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                        word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                    if not similar_words.empty:
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]
                    else:
                        word = word.lower()
                        similar_words = result_similarity['오염문장'].apply(lambda x: difflib.get_close_matches(
                            word, x.split(), n=1, cutoff=0.85)).apply(pd.Series, dtype=object)
                        # 유사한 단어를 가진 행을 찾습니다.
                        matching_rows = result_similarity[similar_words.notna().any(
                            axis=1)]

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
                                result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                        '문장위치'], 'score': result_rows['score'], '장소명': room_en_rows, '기기명': mc_name_rows, '장소표현': room_num_rows, '기기표현': word})
                            else:
                                room_ko_rows = room_ko[indexes]
                                if isinstance(room_ko_rows, str):
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                            '문장위치'], 'score': result_rows['score'], '장소명': room_ko_rows, '기기명': mc_name_rows, '장소표현': room_num_rows, '기기표현': word})
                                else:
                                    result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                            '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': mc_name_rows, '장소표현': mc_room_rows, '기기표현': word})
                        else:
                            result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                    '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': mc_name_rows, '장소표현': "", '기기표현': word})
                    else:
                        result_line_list.append({'파일이름': paper_name, '오염문장': result_rows['오염문장'], '페이지': result_rows['페이지'], '문장위치': result_rows[
                                                '문장위치'], 'score': result_rows['score'], '장소명': "", '기기명': mc_name_rows, '장소표현': "", '기기표현': word})

        # 그 외 문장들 추가
        present_sentence = [d['오염문장'] for d in result_line_list if '오염문장' in d]
        for i in range(len(result_similarity)):
            if result_similarity['오염문장'][i] not in present_sentence:
                result_line_list.append({'파일이름': paper_name, '오염문장': result_similarity['오염문장'][i], '페이지': result_similarity['페이지'][
                                        i], '문장위치': result_similarity['문장위치'][i], 'score': result_similarity['score'][i], '장소명': "", '기기명': "", '장소표현': "", '기기표현': ""})

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
