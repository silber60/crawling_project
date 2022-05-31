import pandas as pd
import re
import time
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException


# category = ['테크·가전', '패션·잡화', '뷰티', '푸드', '홈·리빙', '여행·레저', '스포츠·모빌리티', '출판'] # 8개
import time
from selenium import webdriver
driver = webdriver.Chrome(executable_path= r"./chromedriver.exe")
driver.get("https://www.wadiz.kr/web/wreward/category/296?keyword=&endYn=ALL&order=recent")

time.sleep(2)

try:
    for i in range(1):
        button = driver.find_element_by_xpath(
            f'//*[@id="main-app"]/div[2]/div/div[5]/div[2]/div[2]/div/button')  # 더보기버튼 xpath
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", button)  # click()으로 에러가나서 써줌
        print('page:', i)

except:
    button = driver.find_element_by_class_name
    button.click()
    print('끝남')

table = driver.find_element_by_class_name('ProjectCardList_container__3Y14k') #표 전체
rows = table.find_elements_by_class_name("ProjectCardList_item__1owJa")

wadiz_title = []
results_reward = []
category = []

for index, value in enumerate(rows):  #enumerate는 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능
    title=value.find_element_by_class_name("CommonCard_title__1oKJY")
    result=value.find_element_by_class_name("RewardProjectCard_percent__3TW4_")
    wadiz_title.append(title.text)
    results_reward.append(result.text)
    category.append("여행·레저")

    # print(title.text, result.text)
    time.sleep(0.3)


# df1 = pd.DataFrame({'title':wadiz_title, 'category':category})
df1 = pd.DataFrame({'title':wadiz_title, 'category':category, 'reward':results_reward})
print(len(df1))
import csv
df1.to_csv("./예측데이터/wadiz_여행·레저_달성률_예측데이터.csv".format(len(df1)), mode='w',encoding='utf-8-sig', index=False)