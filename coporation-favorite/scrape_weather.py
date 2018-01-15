import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import OrderedDict
import re
import numpy as np

def extract_content(td):
    content = re.findall(r'\w|\.', td.get_text())
    digits = ''.join([x for x in content if x.isdigit() or x == '.'])
    words = ''.join([x for x in content if x.isalpha()])
    content = (digits + " " + words).strip()
    return content
#scrape on website https://www.wunderground.com/history
def scrape_ecuador_weather(m, d, c, cities):
    filename1 = "https://www.wunderground.com/history/airport/" + cities[c] +"/2017/"
    month = m
    days = d
    filename2 = "/DailyHistory.html"
    pagename = filename1 + str(month) + "/" + str(days) + filename2
    page = requests.get(pagename)
    if page.status_code != 200:
        print("not found!!!   " + str(month) + "/" + str(days))
    soup = BeautifulSoup(page.content, 'html.parser')
    tables = soup.find_all('div', attrs={"id": "observations_details"})
    #TODO: head
    date = "2017-"+ str(month).rjust(2,"0")+ "-"+ str(days).rjust(2,"0")

    #update table title
    table_title = tables[0].find_all('thead')[0]

    titles = []
    for th in tables[0].find_all('thead')[0].find_all('th'):
        titles.append(th.get_text())

    table_body = tables[0].find_all('tbody')
    if len(table_body) != 0:
        table_body = table_body[0]
    else:
        print("no data is found on " + str(month) + "/" + str(days))
        return
    contents = [[] for _ in range(len(titles))]
    for tr in table_body.find_all('tr'): #for each row in table
        tds = tr.find_all('td')
        for i in range(len(tds)): # for each entry in row
            content = extract_content(tds[i])
            try:
                contents[i].append(content)
            except:
                print(pagename)
    df = OrderedDict()
    df["City"] = [c for _ in range(len(contents[0]))]
    df["Date"] = [date for _ in range(len(contents[0]))]
    for title, content in zip(titles, contents):
        df[title] = content
    return pd.DataFrame.from_dict(df).drop_duplicates(subset = [titles[0]], keep=False)


titles = ['Time',
          'Temp',
          'Dew_Point',
          'Humidity',
          'Pressure',
          'Visibility',
          'Wind_Dir',
          'Wind_Speed',
          'Gust_Speed',
          'Precip',
          'Events',
          'Conditions']
 
 cities = {"Latacunga": "SELT",
          "Ambato": "SEAM",
          "Cuenca": "SECU",
          "Salinas": "SESA",
          "Guayaquil": "SEGU",
         "Loja": "wmo/84270",
         "Machala": "SERO"}
         
months = [1,2,3,4,5,6,7,8]
days = range(1,32)
dataset = None
for city in cities:
    for month in months:
        for day in range(1,32):
            data = scrape_ecuador_weather(month, day, city, cities)
            if dataset is None:
                dataset = data
            else:
                dataset = pd.concat([dataset, data])
dataset.to_csv('weather.csv',index = False)

#other cities not yet finished
"""
weather_cities = ['Ambato', 'Cuenca', 'Guayaquil', 'Loja', 'Machala', 'Manta', 'Quito']
other_cities = {"Salinas": "Guayaquil", "Liberated": "Guayaquil", "Cayambe": "Ibarra", "Babahoyo": "Guayaquil",
               "Santo Domingo": "Quito", "Riobama": "Ambato", "Plays": "Machala", "Daule": "Guayaquil",
               ""}
'El Carmen','Esmeraldas', 'Guaranda',  'Playas', 'Quevedo',
"""
