import asyncio
import time
import aiohttp
import aiomysql
from fake_useragent import UserAgent
from lxml import etree
import logging
import pandas as pd

logging.basicConfig(filename='logger.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
start = time.time()


async def get_request(url, semaphore):
    async with semaphore:
        timeout = aiohttp.ClientTimeout(total=None, connect=None)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            ua = UserAgent()
            headers = {"User-Agent": ua.random}
            async with await sess.get(url, headers=headers) as response:
                await asyncio.sleep(0.0001)
                page_text = await response.text()
                result = await parse(page_text, url.split("/")[-1])
                logging.info("{} finished".format(url.split("/")[-1]))
            return result


async def parse(page_text, word):
    pagecontent = etree.HTML(page_text)
    try:
        trendchart = pagecontent.xpath("//*[@id='trendingWordsFrequency']")[0].text
        maxyear = trendchart.split(";")[0]
        trendvalue = trendchart.split(";")[1]
        trend = (maxyear, trendvalue)
    except IndexError as e:
        maxyear, trendvalue, trend = 'NA', 'NA', 'NA'
    await writedb(word, maxyear, trendvalue)
    return trend


async def writedb(word, maxyear, trendvalue):
    pool = await aiomysql.create_pool(host='localhost', port=3306,
                                      user='root', db='datascience')
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute('insert into raw_eng_dict (word,maxyear,frequency) values ("{}","{}","{}")'.
                              format(word, maxyear, trendvalue))
            await conn.commit()
    pool.close()
    await pool.wait_closed()


async def main():
    task_list = []
    semaphore = asyncio.Semaphore(50)
    for i in range(len(englishword)):
        url = "https://www.collinsdictionary.com/dictionary/english/{0}".format(englishword.loc[i, 0])
        async_task = asyncio.create_task(get_request(url, semaphore), name=englishword.loc[i, 0])
        task_list.append(async_task)
    done,pending = await asyncio.wait(task_list)
    return done,pending


englishword = pd.read_table('englishwords.txt', header=None)
done = asyncio.run(main())
print('总耗时:', time.time() - start)
# pd.Series(list(done)).to_csv('aa.csv')
