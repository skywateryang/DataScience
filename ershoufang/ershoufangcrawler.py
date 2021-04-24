import logging
import pyodbc
import re
import time
from datetime import datetime

import requests
from lxml import etree

logger = logging.getLogger()

file_handler = logging.FileHandler('C:\\Users\\skywater\\PycharmProjects\\personal\\log.txt', encoding='UTF-8',
                                   mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome \
                      /73.0.3683.103 Safari/537.36',
           'Referer': 'https://sh.lianjia.com/ershoufang/',
           'Accept': 'application/json, text/javascript, */*; q=0.01'}


def get_houseurl(url):
    try:
        time.sleep(0.1)
        referer = url[0:url.rfind('/')]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome \
                              /73.0.3683.103 Safari/537.36',
                   'Referer': '%s' % (url),
                   'Accept': 'application/json, text/javascript, */*; q=0.01'}
        pagetext = requests.get(url, timeout=(6, 8), headers=headers)
        logging.info(pagetext.status_code)
        pagehtml = etree.HTML(pagetext.text)
        houseurllist = pagehtml.xpath(
            '//li[contains(@class,"clear") and contains(@class,"LOGCLICKDATA")]/div[1]/div[1]/a/@href')
        logging.info(houseurllist)
        for houseurl in houseurllist:
            logging.info(houseurl)
            get_houseinfo(houseurl)
    except requests.exceptions.RequestException as e:
        logging.info(url + 'timeout error')


def get_houseinfo(url):
    try:
        # time.sleep(0.1)
        urllink = url
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome \
                                      /73.0.3683.103 Safari/537.36',
                   'Referer': '%s' % (url),
                   'Accept': 'application/json, text/javascript, */*; q=0.01'}
        housepage = requests.get(url, timeout=(6, 8), headers=headers)
        if housepage.status_code == 200:
            househtml = etree.HTML(housepage.text)
            try:
                district = househtml.xpath('/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[1]/text()')[0]
                title = househtml.xpath('/html/body/div[3]/div/div/div[1]/h1/@title')[0]
                totalvalue = househtml.xpath('/html/body/div[5]/div[2]/div[2]/span[1]/text()')[0] + \
                             househtml.xpath('/html/body/div[5]/div[2]/div[2]/span[2]/span[1]/text()')[0]
                unitprice = househtml.xpath('/html/body/div[5]/div[2]/div[2]/div[1]/div[1]/span/text()')[0]
                communityname = househtml.xpath('/html/body/div[5]/div[2]/div[4]/div[1]/a[1]/text()')[0]
                areaname = househtml.xpath('/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[2]/text()')[0]
                housetype = househtml.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[1]/text()')[0]
                size = househtml.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[3]/text()')[0]
                orientation = househtml.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[7]/text()')[0]
                floor = househtml.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[2]/text()')[0]
                try:
                    lift_household_ratio = \
                        househtml.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[10]/text()')[0]
                except IndexError as e:
                    lift_household_ratio = ''
                    orientation = househtml.xpath('//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[5]/text()')[0]
                listtime = househtml.xpath('//*[@id="introduction"]/div/div/div[2]/div[2]/ul/li[1]/span[2]/text()')[0]
                lasttradetime = \
                househtml.xpath('//*[@id="introduction"]/div/div/div[2]/div[2]/ul/li[3]/span[2]/text()')[0]
            except IndexError:
                logging.info('info find error')
                return
            recorddate = datetime.now()
            # builddate=househtml.xpath('//*[@id="resblockCardContainer"]/div/div/div[2]/div/div[2]/span/text()')[0]
            conn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;DATABASE=mydb;UID=sa;PWD=123456')
            cursor = conn.cursor()
            sql = "insert into secondhandhouse(title,district,totalvalue,unitprice,communityname,areaname,housetype,size, \
            orientation,floor,lift_household_ratio,listtime,lasttradetime,recorddate) values('{0}', '{1}',  \
            '{2}', {3}, '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}', '{12}', CURRENT_TIMESTAMP)".format(
                title, district, totalvalue, unitprice, communityname, areaname, housetype, size,
                orientation, floor, lift_household_ratio, listtime, lasttradetime)
            try:
                cursor.execute(sql)
                conn.commit()
                logging.info(str(title) + '   write success')
            except Exception as e:
                logging.info(urllink + " insert error")
                conn.rollback()
            finally:
                cursor.close()
                conn.close()
        else:
            logging.info(url + str(housepage.status_code) + 'status code error')
    except requests.exceptions.RequestException as e:
        logging.info(url + 'timeout error')


if __name__ == '__main__':
    # get arealist
    districtlist = ['pudong', 'minhang', 'baoshan', 'xuhui', 'putuo', 'yangpu', 'changning', 'songjiang', 'jiading',
                    'huangpu', 'jingan', 'zhabei', 'hongkou', 'qingpu', 'fengxian', 'jinshan', 'chongming']
    arealist = []
    for districtname in districtlist:
        districturl = 'https://sh.lianjia.com/ershoufang/' + districtname + '/'
        time.sleep(0.1)
        districtpage = requests.get(districturl, headers=headers)
        districtpagehtml = etree.HTML(districtpage.text)
        areaname = districtpagehtml.xpath('/html/body/div[3]/div/div[1]/dl[2]/dd/div[1]/div[2]//@href')
        arealist = arealist + areaname
    arealistloop = list(set(arealist))
    logging.info(arealistloop)
    logging.info('get area success')

    areafinish = []
    for area in arealistloop:
        if (area not in areafinish) and (area.split('/')[2] not in districtlist):
            logging.info('begin to query ' + area)
            areaurl = 'https://sh.lianjia.com' + area
            time.sleep(0.1)
            homepage_by_area = requests.get(areaurl, headers=headers)
            homepage_by_area_html = etree.HTML(homepage_by_area.text)
            try:
                pagecount = \
                    re.split("[,:]",
                             homepage_by_area_html.xpath('//*[@id="content"]/div[1]/div[8]/div[2]/div/@page-data')[0])[
                        1]
            except IndexError as e:
                pagecount = 1
                logging.info(area + ' only 1 page')
            for page in range(1, int(pagecount) + 1):
                logging.info('begin to search' + area + str(page))
                pageurl = areaurl + '/pg' + str(page)
                get_houseurl(pageurl)
                logging.info(area + str(page) + ' finished')
