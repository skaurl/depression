from selenium import webdriver
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import re
from tqdm import tqdm

df = pd.read_csv('blog text data.csv', encoding='cp949')

driver = webdriver.Chrome('/Users/kimnamhyeok/Google 드라이브/chromedriver')

url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=네이버+맞춤법+검사기&oquery=네이버+맞춤법+검사기&tqi=Uoxwjdp0JXoss7zkMPhssssstiR-136322'
driver.get(url)

for i in tqdm(range(len(df))):
    driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[1]/textarea').clear()
    #맞춤법 검사기의 입력창을 초기화

    driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[1]/textarea').send_keys('\n' + df.iloc[i, 3])
    #맞춤법 검사기의 입력창에 문장을 입력

    driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[2]/button').click()
    #맞춤법 검사기의 '검사하기' 버튼을 클릭

    time.sleep(0.1)
    #맞춤법 검사에 걸리는 시간을 0.1이내로 상정
    #만약 0.1초 이내에 맞춤법 검사가 안될 경우, 0.5초 그리고 1초로 맞춤법 검사

    soup = bs(driver.page_source, 'html.parser')
    A = soup.select('div.text_area')[1].select('p')[0].get_text()
    #맞춤법 검사를 마친 문장을 크롤링

    A = re.sub('^맞춤법 검사를 원하는 단어나 문장을 입력해 주세요.', '', A)
    #문장 앞에 딸려오는 '맞춤법 검사를 원하는 단어나 문장을 입력해 주세요.'는 생략

    if A == '맞춤법 검사 중입니다. 잠시만 기다려주세요.':
        #0.1초로 맞춤법 검사가 안될 경우, 0.5초로 맞춤법 검사를 반복
        driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[1]/textarea').clear()
        driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[1]/textarea').send_keys('\n' + df.iloc[i, 3])
        driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[2]/button').click()
        time.sleep(0.5)
        soup = bs(driver.page_source, 'html.parser')
        A = soup.select('div.text_area')[1].select('p')[0].get_text()
        A = re.sub('^맞춤법 검사를 원하는 단어나 문장을 입력해 주세요.', '', A)

    if A == '맞춤법 검사 중입니다. 잠시만 기다려주세요.':
        #0.5초로 맞춤법 검사가 안될 경우, 1초로 맞춤법 검사를 반복
        driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[1]/textarea').clear()
        driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[1]/textarea').send_keys('\n' + df.iloc[i, 3])
        driver.find_element_by_xpath('//*[@id="grammar_checker"]/div/div[2]/div[1]/div[1]/div/div[2]/button').click()
        time.sleep(1)
        soup = bs(driver.page_source, 'html.parser')
        A = soup.select('div.text_area')[1].select('p')[0].get_text()
        A = re.sub('^맞춤법 검사를 원하는 단어나 문장을 입력해 주세요.', '', A)

    A = ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(A)).split())
    #문장에서 숫자, 영어, 한글만 추출

    df.iloc[i, 3] = A

df.to_csv('dataset.csv', index=False, encoding='cp949')
driver.quit()
