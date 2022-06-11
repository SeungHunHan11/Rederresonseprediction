import requests
import selenium
import os
import pandas as pd
import selenium
from selenium.webdriver import *
from selenium import webdriver
import time
import sys
from bs4 import BeautifulSoup
import urllib.request
import requests
from lxml import etree, html
from tqdm import tqdm
chrome_driver = 'C:/chromedriver.exe'



def content_crawling(link_list,date,min):
    #options=webdriver.ChromeOptions()
    #options.add_argument("--window-position=4150,0")
    #options.add_argument("--window-size=1000,600")
    driver = webdriver.Chrome(chrome_driver)

    news_data_set={'Date':[],'Title':[],'Total_reaction':[],'Good':[],'Bad':[]}
    for news in tqdm(link_list):
        try:
            driver.get(news)
            reaction = int((driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div[2]/div[1]/div[1]/a/span[3]').text).replace(',',''))
            if reaction >= int(min):
                news_data_set['Total_reaction'].append(reaction)
                news_data_set['Date'].append(date)
                news_data_set['Title'].append(
                    driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div[1]/h4').text)
                time.sleep(0.5)
                driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div[2]/div[1]/div[1]/a/span[3]').click()
                time.sleep(0.5)
                reaction_list = driver.find_elements_by_xpath(
                    '//*[@id="content"]/div/div[1]/div/div[2]/div[1]/div[1]/ul/li/a/span[2]')
                reaction_list = [x.text for x in reaction_list]
                reaction_list= [x.replace(',','') for x in reaction_list]
                reaction_list = [int(x) for x in reaction_list]
                news_data_set['Good'].append(reaction_list[0] + reaction_list[3] + reaction_list[4])
                news_data_set['Bad'].append(reaction_list[1] + reaction_list[2])
            else:
                continue
        except:
            continue
    driver.close()
    return pd.DataFrame(news_data_set)

def crawling(head,start_date, end_date, min_reaction=100,verbose=True):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = pd.date_range(start_date, end_date, freq='D').strftime('%Y%m%d')
    news_data_set={'Date':[],'Title':[],'Total_reaction':[],'Good':[],'Bad':[]}
    contents = pd.DataFrame()
    driver = webdriver.Chrome(chrome_driver)

    for D in tqdm(dates):
        page_num=1


        url=head+str(page_num)+'&date='+str(D)+'&isphoto=N'
        driver.get(url)

        while True:

            total_linklist = []

            time.sleep(0.3)

            try:
                nextb = driver.find_element_by_css_selector('#_pageList > a.next')
            except:
                nextb = None

            if nextb !=None:
                link_html = driver.find_elements_by_xpath('//*[@id="_newsList"]/ul/li/div/a')
                total_linklist.append([x.get_attribute('href') for x in link_html])
                content=content_crawling(link_list=total_linklist[0], date=D,min=min_reaction)

                contents=pd.concat([contents,content])

                page_num+=1
                url = head + str(page_num) + '&date=' + str(D) + '&isphoto=N'
                driver.get(url)

            elif nextb==None:

                page=driver.find_elements_by_xpath('//*[@id="_pageList"]')
                page_length=sum([x.isnumeric() for x in [x.text for x in page][0].split()])

                for i in range(int(page_length)):

                    if i!=0:
                        url = head + str(page_num) + '&date=' + str(D) + '&isphoto=N'
                        driver.get(url)

                    time.sleep(0.5)

                    total_linklist = []
                    link_html = driver.find_elements_by_xpath('//*[@id="_newsList"]/ul/li/div/a')
                    total_linklist.append([x.get_attribute('href') for x in link_html])

                    content = content_crawling(link_list=total_linklist[0],date=D,min=min_reaction)
                    contents = pd.concat([contents, content])

                    print(page_num - 1, 'Page Done')

                    if verbose == True:
                        print(content.head(5))

                    print('Total', len(contents), " articles collected")
                    page_num+=1


                break

            print(page_num-1,'Page Done')

            if verbose==True:
                print(content.head(5))

            print('Total',len(contents)," articles collected")
            contents.to_pickle('News_Conpilation.pkl')

    return contents