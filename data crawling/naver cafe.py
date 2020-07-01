from selenium import webdriver
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import random
from tqdm import tqdm

driver = webdriver.Chrome('/Users/kimnamhyeok/Google 드라이브/chromedriver')
driver.get('https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com')
time.sleep(10)
#네이버 자동화 입력을 피하기 위해 10초 안에 직접 로그인

cafe_df_all = pd.DataFrame()

cafe_url = ''
#크롤링하고자 하는 네이버 카페의 게시판 주소
#url은 비공개 처리

for page in tqdm(range(1,1001,1)):
    #최대 1000페이지까지 크롤링이 가능

    driver.get(cafe_url+str(page))
    driver.switch_to.frame('cafe_main')

    article_list = driver.find_elements_by_css_selector('div.inner_list > a.article')
    article_urls = [ i.get_attribute('href') for i in article_list ]
    #페이지 상의 게시판에 있는 게시글의 url을 수집

    res_list = []

    for article in article_urls:
        time.sleep(random.random())
        driver.get(article)
        driver.switch_to.frame('cafe_main')
        soup = bs(driver.page_source, 'html.parser')

        article_id = soup.select('td.p-nick')[0].select('a')[0].get_text()
        #작성자의 아이디

        title = soup.select('div.tit-box span.b')[0].get_text()
        #게시글의 제목

        content_tags = soup.select('#tbody')[0].select('p')
        #게시글의 내용

        article_date = soup.select('td.m-tcol-c.date')[0].get_text()
        #게시글의 작성 일자

        for content in range(len(content_tags)):
            #\n 단위로 잘라서 게시글의 내용을 수집
            res_list.append({'id':article_id, 'url' : article, 'title' : title, 'content' : content_tags[content].get_text(), 'date':article_date})

    cafe_df = pd.DataFrame(res_list)
    cafe_df_all = pd.concat([cafe_df_all,cafe_df], axis = 0)

cafe_df_all.to_pickle('naver_cafe.pickle')