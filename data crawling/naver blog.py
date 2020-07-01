from selenium import webdriver
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import re
import time
import os

driver = webdriver.Chrome('/Users/kimnamhyeok/Google 드라이브/chromedriver')

blog_df_all = pd.DataFrame()

for i in range(101):
    #최대 100페이지까지 크롤링이 가능

    try:
        time.sleep(1)
        driver.get('https://search.naver.com/search.naver?date_from=20180101&date_option=8&date_to=20190101&dup_remove=1&nso=a%3At%2Cp%3Afrom20180101to20190101&post_blogurl=&post_blogurl_without=&query=일상글&sm=tab_pge&srchby=title&st=sim&where=post&start='+str(i*10+1))
        #2018/01/01~2019/01/01 '일상글' 키워드로 검색된 블로그 게시글을 크롤링

        blog_urls = []

        for j in range(1,11,1):
            #한 페이지에 10개의 게시글이 존재
            blog_url = driver.find_elements_by_xpath('//*[@id="sp_blog_'+str(j)+'"]/dl/dt/a')[0].get_attribute('href')
            blog_urls.append(blog_url)
            #페이지 상의 블로그 url을 수집

        res_list = []

        for j in blog_urls:
            try:
                time.sleep(1)
                driver.get(j)
                driver.switch_to.frame('mainFrame')
                soup = bs(driver.page_source, 'html.parser')

                #블로그는 html 형식이 일정하지 않으므로, 가장 많이 쓰이고 있는 html 형식을 기준으로 코드를 작성
                #또한, 특정 년도를 기준으로 블로그에서 가장 많이 쓰이고 있는 html 형식이 변경되기 때문에 이 점에 유의하여 코드를 변경

                try:
                    id = soup.find('span', {'class': 'nick'}).get_text()
                except:
                    id = ''
                #작성자의 아이디

                try:
                    title = soup.find('div', {'class': 'se-module se-module-text se-title-text'}).get_text()
                    title = ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(title.strip())).split())
                    #숫자, 영어, 한글만 추출
                except:
                    title = ''
                #게시글의 제목

                try:
                    date = soup.find('span', {'class': 'se_publishDate pcol2'}).get_text()
                except:
                    date = ''
                #게시글의 작성 일자

                content = soup.find_all('div', {'class': 'se-section se-section-text se-l-default'})
                #게시글의 내용

                for k in range(len(content)):
                    #\n 단위로 잘라서 게시글의 내용을 수집
                    try:
                        sentence = content[k].find_all('span',{'class':''})
                        for l in range(len(sentence)):
                            sentence[l] = ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(sentence[l].get_text().strip())).split())
                            #숫자, 영어, 한글만 추출
                            res_list.append({'id':id, 'url' : j, 'title' : title, 'content' : sentence[l], 'date':date})
                    except:
                        pass
            except:
                pass
            blog_df = pd.DataFrame(res_list)
            blog_df_all = pd.concat([blog_df_all, blog_df], axis=0)
    except:
        pass

blog_df_all[blog_df_all['content'] == ''] = np.nan
#빈 데이터를 nan으로 변경

blog_df_all = blog_df_all.dropna(how='any')
#nan이 있는 행을 제거

blog_df_all = blog_df_all.drop_duplicates()
#중복 제거

blog_df_all = blog_df_all.reset_index()
del blog_df_all['index']
#index 재정렬

blog_df_all['date'] = pd.to_datetime(blog_df_all['date'])
#date 열을 datetime형으로 변경

blog_df_all.to_csv('naver_blog.csv', index=False, encoding='cp949')

driver.quit()