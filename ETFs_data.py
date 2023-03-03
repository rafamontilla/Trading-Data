from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


#Acess the etf.com 

driver = webdriver.Chrome(executable_path='/home/gabriela/Downloads/chromedriver')

url = 'https://www.etf.com/etfanalytics/etf-finder'

driver.get(url)

#Awaits the site to fully load
time.sleep(5)

#Clicks on button to display 100 ETFs
button_100 = driver.find_element('xpath', '''/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/section[2]/div[1]/div/div[4]/button/label/span''')

driver.execute_script('arguments[0].click();',button_100)

#Find the number of pages needed to iterate and retrieve the data
pages_number = driver.find_element('xpath', '''/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/section[2]/div[2]/div/label[2]''')

pages_number = pages_number.text.replace("of ","")

pages_number = int(pages_number)

#Iterate over the table
tables_for_page = []

for page in range(0, pages_number):
    
    table = driver.find_element('xpath', '''/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/div/table''')

    html_table = table.get_attribute('outerHTML')

    final_table = pd.read_html(html_table)[0]

    tables_for_page.append(final_table)

    button_next_page = driver.find_element('xpath', '/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/section[2]/div[2]/div/span[2]') 

    driver.execute_script('arguments[0].click();',button_next_page)

fund_basics_data = pd.concat(tables_for_page)

'''print(fund_basics_data)'''

#search data by performance
button_performance = driver.find_element('xpath', '''/html/body/div[5]/
section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/
ul/li[2]/span''')

driver.execute_script('arguments[0].click();',button_performance)

for page in range(0, pages_number):
    
    button_back_page = driver.find_element('xpath', '''/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/section[2]/div[2]/div/span[1]''')

    driver.execute_script('arguments[0].click();', button_back_page)


tables_for_page = []

for page in range(0, pages_number):
    
    table = driver.find_element('xpath', '''/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/div/table''')

    html_table = table.get_attribute('outerHTML')

    final_table = pd.read_html(html_table)[0]

    tables_for_page.append(final_table)

    button_next_page = driver.find_element('xpath', '/html/body/div[5]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/section[2]/div[2]/div/span[2]') 

    driver.execute_script('arguments[0].click();',button_next_page)

performance_data = pd.concat(tables_for_page)

'''print(performance_data)'''

#Make the final table
fund_basics_data = fund_basics_data.set_index('Ticker')

'''print(fund_basics_data)'''

performance_data = performance_data.set_index('Ticker')

performance_data = performance_data [['1 Year', '5 Years', '10 Years']]

'''print(performance_data)'''

final_data = fund_basics_data.join(performance_data)

print(final_data)
