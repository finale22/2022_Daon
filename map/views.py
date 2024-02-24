from django.http import HttpResponse
from django.shortcuts import render
from konlpy.tag import Okt
import json
from django import forms


import pandas as pd
import os
import platform
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer

# 모듈 import
import re
import pandas as pd
from tqdm import tqdm

from collections import Counter
# from pykospacing import Spacing #pip install git+https://github.com/haven-jeon/PyKoSpacing.git
import time


from django.views.decorators.csrf import csrf_exempt



from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
property_list = ["hash"]





def home(request):


    return render(request, 'map/home.html')


def main(request):
    db, ori_db =  read_category()



    return render(request, 'map/main.html', db)


@csrf_exempt
def recomand(request):
    db = {}
    ori_data, df =read_data()
    df = df.to_dict('records')

    if request.method == 'GET':  # GET 메소드로 값이 넘어 왔다면,
        for key in property_list:
            # 값이 넘어 오지 않았다면 "", 값이 넘어 왔다면 해당하는 값을 db에 넣어줌
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""  # 삼항 연산자


        hash_tag = request.GET
        hash_dict = hash_tag.dict()


        if hash_dict=={}:
            db['dataframe'] = df
            db['ori_data'] = ori_data
        else:
            value = list(hash_dict.values())[0]
            hash_df = read_hash(value)
            hash_df = hash_df.to_dict('records')
            db['dataframe'] = hash_df

    elif request.method == 'POST':
        mention = request.POST.get('want')
        df = main_func(mention)
        df = df.to_dict('records')
        db['dataframe'] = df




    return  render(request, 'map/recomand.html', db)

def make(request):

    return render(request, 'map/make.html')


##불러오기 기능

def read_data():
    data = pd.read_csv('C:/github/ICT/ict_project/daon/daon/data/db.csv', encoding='ms949')
    df=pd.DataFrame()
    df['사업명']=data['사업명']
    df['지역']=data['지자체']
    df['URL']=data['URL']
    return(data, df)

def read_category():
    index_1 = [3,27,14]
    index_2 = [10, 11]
    index_3 = [7, 15, 23]
    index_4 = [14, 16, 19]
    index_5 = [16, 20,21]
    num = [1,2,3]

    df_index = ['사업명', '지자체', '해시태그', 'URL']

    db = {}
    df, _ = read_data()
    ori_db = pd.DataFrame()
    for i in range(len(df_index)):
        ori_db[df_index[i]] = df[df_index[i]]

    for i in range(3):
        db[f'latest_{num[i]}'] = ori_db.loc[index_1[i]][df_index[0]]
        db[f'city_{num[i]}'] = ori_db.loc[index_1[i]][df_index[1]]
        db[f'hash_{num[i]}_1'] = ori_db.loc[index_1[i]][df_index[2]]
        db[f'link_{num[i]}'] = ori_db.loc[index_1[i]][df_index[3]]

    for i in range(2):
        db[f'baby_{num[i]}'] = ori_db.loc[index_2[i]][df_index[0]]
        db[f'b_city_{num[i]}'] = ori_db.loc[index_2[i]][df_index[1]]
        db[f'b_hash_{num[i]}_1'] = ori_db.loc[index_2[i]][df_index[2]]
        db[f'b_link_{num[i]}'] = ori_db.loc[index_2[i]][df_index[3]]

    for i in range(3):
        db[f'for_you_{num[i]}'] = ori_db.loc[index_3[i]][df_index[0]]
        db[f'f_city_{num[i]}'] = ori_db.loc[index_3[i]][df_index[1]]
        db[f'f_hash_{num[i]}'] = ori_db.loc[index_3[i]][df_index[2]]
        db[f'f_link_{num[i]}'] = ori_db.loc[index_3[i]][df_index[3]]

    for i in range(3):
        db[f'real_time_{num[i]}'] = ori_db.loc[index_4[i]][df_index[0]]
        db[f'r_city_{num[i]}'] = ori_db.loc[index_4[i]][df_index[1]]
        db[f'r_hash_{num[i]}'] = ori_db.loc[index_4[i]][df_index[2]]
        db[f'r_link_{num[i]}'] = ori_db.loc[index_4[i]][df_index[3]]

    for i in range(3):
        db[f'younger_{num[i]}'] = ori_db.loc[index_5[i]][df_index[0]]
        db[f'y_city_{num[i]}'] = ori_db.loc[index_5[i]][df_index[1]]
        db[f'y_hash_{num[i]}'] = ori_db.loc[index_5[i]][df_index[2]]
        db[f'y_link_{num[i]}'] = ori_db.loc[index_5[i]][df_index[3]]

    return db, ori_db

def read_hash(value):
    hash_key = value
    df_index = ['사업명', '지자체', '해시태그', 'URL']


    df, _ = read_data()
    ori_db = pd.DataFrame()
    for i in range(len(df_index)):
        ori_db[df_index[i]] = df[df_index[i]]

    ori_db = ori_db[ori_db['해시태그']==value]
    ori_db['지역'] = ori_db['지자체']

    return ori_db



    
"""-------------------------------------------------------------------------------------------------------------------"""


# 빈도분석 + 키워드 추출 함수
def load_area():
    df_body = df.iloc[:, 1:2]
    body = df_body['사업명'] + " " + df['기타']
    return body


def load_stopwords():
    with open('C:/github/ICT/ict_project/daon/daon/data/stopwords.txt', 'r') as f:
        list_file = f.readlines()
    return list_file[0].split(",")


# 정규 표현식을 통해 한글 단어만 남기기 (이모티콘, 초성, 영어 제거)
def extract_word(text):
    hangul = re.compile('[^가-힣]')
    result = hangul.sub(' ', text)
    return result


def make_wordlist(body, stopwords):
    result = []
    for i in body:
        # 정규표현식 적용
        # print("데이터 정제 중....")
        words = pd.Series(i)
        words = words.apply(lambda x: extract_word(x))
        # spacing = Spacing()
        # words = words.apply(lambda x:spacing(x))

        # 형태소 추출
        # print("형태소 추출 중....")
        okt = Okt()
        words = " ".join(words.tolist())
        words = okt.morphs(words, stem=True)

        # 한글자 제거
        # print("한글자 제거 중....")
        words = [x for x in words if len(x) > 1]

        # 불용어 제거
        # print("불용어 제거 중....")
        words = [x for x in words if x not in stopwords]

        # 형태소 분석 결과를 리스트에 저장
        result.append(words)

    return result


def run():
    body = load_area()
    listplcy = make_wordlist(body, load_stopwords())
    return listplcy


def run2(txt):
    return make_wordlist(txt, load_stopwords())


def prt_cos(plcy, scan):  # 코사인 유사도 연산하고 출력
    # 가장 유사한 선택지
    max_cos = 0
    max_tmp = -1
    # 두번째 선택지
    sec_cos = 0
    sec_tmp = -1
    # 세번째 선택지
    thir_cos = 0
    thir_tmp = -1

    for i in range(len(plcy)):
        sentences = (plcy[i], scan)
        tfidf_vectorizer = TfidfVectorizer()

        # 문장 벡터화 하기(사전 만들기)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # 코사인 유사도
        from sklearn.metrics.pairwise import cosine_similarity

        # 첫번째와 두번째 문장 비교
        cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        if max_cos < cos_similar:  # 최고 유사 코드 바뀌었을 때
            if (government == 'default') | (df['지자체'][i] == government):  # 지자체가 맞다면 뒷 내용 실행, 아니면 무시, 기본값일땐 다 출력
                if max_tmp == -1:  # 처음 바뀌었을 때
                    max_cos = cos_similar
                    max_tmp = i
                elif sec_tmp == -1:  # 두번째 바뀌었을 때
                    sec_cos = max_cos
                    sec_tmp = max_tmp
                    max_cos = cos_similar
                    max_tmp = i
                else:  # 3번째 까지 설정되었을 때
                    thir_cos = sec_cos
                    thir_tmp = sec_tmp
                    sec_cos = max_cos
                    sec_tmp = max_tmp
                    max_cos = cos_similar
                    max_tmp = i

    if max_cos > 0.12:
        print("연관성 가장 높은 사업은 \"{}\"입니다.__ 점수 : {}".format(df['사업명'][max_tmp], max_cos))
        max_lst = [df['사업명'][max_tmp], df['지자체'][max_tmp], df['URL'][max_tmp]]
        if sec_cos > 0.11:
            print("연관성 2번째 높은 사업은 \"{}\"입니다.__ 점수 : {}".format(df['사업명'][sec_tmp], sec_cos))
            sec_lst = [df['사업명'][sec_tmp], df['지자체'][sec_tmp], df['URL'][sec_tmp]]
            if thir_cos > 0.1:
                print("연관성 3번째 높은 사업은 \"{}\"입니다.__ 점수 : {}".format(df['사업명'][thir_tmp], thir_cos))
                thir_lst = [df['사업명'][thir_tmp], df['지자체'][thir_tmp], df['URL'][thir_tmp]]
            else:
                thir_tmp = -1
        else:
            sec_tmp = -1
            thir_tmp = -1
    else:
        max_tmp = -1
        sec_tmp = -1
        thir_tmp = -1
        print("알맞은 지원정책이 없습니다. 다시 검색해보세요")

    if max_tmp == -1:
        return pd.DataFrame((['연관된 지원사업이 없습니다.', '', '']), index=['사업명', '지역', 'URL']).T
    elif sec_tmp == -1:
        print("1")
        return pd.DataFrame((zip(max_lst)), index=['사업명', '지역', 'URL']).T
    elif thir_tmp == -1:
        print("2")
        return pd.DataFrame((zip(max_lst, sec_lst)), index=['사업명', '지역', 'URL']).T
    else:
        print("3")
        return pd.DataFrame((zip(max_lst, sec_lst, thir_lst)), index=['사업명', '지역', 'URL']).T


# ==================== 메인 ====================
# 전역변수
df = pd.read_csv('C:/github/ICT/ict_project/daon/daon/data/db.csv', encoding='ms949')
listplcy = run()  # 정책모음집 형태소분석 수행 (미리해 놓을 수도 있음)
plcy = [""] * len(listplcy)

for i in range(len(listplcy)):
    plcy[i] = ",".join(listplcy[i]).replace(',', ' ')
# print(plcy) # 지원사업 형태소분석 리스트 출력

# 검색 문장도 형태소 분석
tmpscan = "검색어"  # 검색어
government = "default"  # 지자체


def main_func(tmpscan):
    global plcy
    global government
    # tmpscan = "정장"            #여기에 정책 요구사항 입력
    government = "default"  # 여기에 지자체 입력

    scan = []
    scan.append(tmpscan)
    scan = ",".join(run2(scan)[0]).replace(',', ' ')

    result_df = prt_cos(plcy, scan)

    return result_df

