import os
import time
import random
import requests
from bs4 import BeautifulSoup
from lib.Utils import GetRoot
from urllib.parse import urlparse, urlunparse, urlencode


# Load environment variables
GetRoot()

# 記錄所有法規內的法條
lis_laws = []

# 取得全國法規資料庫所有法規的連結
url = "https://law.moj.gov.tw/Hot/Hot.aspx"

# Header 資訊
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.178 Safari/537.36"
}

# 發出請求
response = requests.get(url, headers=headers)

# 抓取熱門法規超連結
dict_law = {}
if response.status_code == 200:
    # 解析 HTML
    soup = BeautifulSoup(response.text, "lxml")
    a_tags = soup.find_all("a")
    for a_tag in a_tags:
        if a_tag.get("id") == "hlkLawName":
            law_url = a_tag.get("href")
            law_name = a_tag.text
            dict_law.setdefault(law_name, law_url)
    print(f">>> 抓取熱門法規數量: {len(dict_law.keys())}")

    # 訪問各項法規的超連結
    for law_name, law_url in dict_law.items():
        # 組合URL
        parsed_url = urlparse(url)

        # 修改路徑
        new_path = "/LawClass/LawAll.aspx"

        # 設置查詢參數
        query_params = {"pcode": str(law_url.split('=')[-1])}
        new_query = urlencode(query_params)

        # 組合新的 URL
        law_url = urlunparse((parsed_url.scheme, parsed_url.netloc, new_path, '', new_query, ''))
        print(f"\t法規名稱: {law_name};\t網址: {law_url}")

        # 隨機等待時間
        wait_time = random.uniform(5, 8)
        time.sleep(wait_time)

        # 發出請求
        response_law = requests.get(law_url, headers=headers)

        if response_law.status_code == 200:
            # 解析 HTML
            soup_law = BeautifulSoup(response_law.text, "lxml")

            # 取得網站擷取的內容
            lis_laws_contents = []

            # 取得法規基本資訊 (名稱)
            law_name = ''
            title = soup_law.find_all("table", class_="table")[0]
            for tr in title.find_all("tr"):
                if tr.find("th").text == "法規名稱：":
                    content = tr.find("th").text + tr.find("a").text
                    law_name = tr.find("a").text

            # 取得法條資訊
            law_content = soup_law.find_all("div", class_="law-reg-content")[0]
            divs = law_content.find_all("div")

            # 初始化後續紀錄資訊的變數避免因為"沒有定義變數"而報錯
            int_item = 0
            chapter = ''
            article = ''
            content = ''
            item_num = ''
            lis_chapter = ["", "", "", "", ""]

            # 遞迴所有"div"標籤
            for div in divs:
                # 取得 class 名稱
                class_list = div.get("class", [])
                class_str = " ".join(class_list) if class_list else "None"

                # 取得章節
                lis_chapter[0] = div.text.strip(' ').replace(' ', '') if class_str == "h3 char-1" else lis_chapter[0]
                lis_chapter[1] = div.text.strip(' ').replace(' ', '') if class_str == "h3 char-2" else lis_chapter[1]
                lis_chapter[2] = div.text.strip(' ').replace(' ', '') if class_str == "h3 char-3" else lis_chapter[2]
                lis_chapter[3] = div.text.strip(' ').replace(' ', '') if class_str == "h3 char-4" else lis_chapter[3]
                lis_chapter[4] = div.text.strip(' ').replace(' ', '') if class_str == "h3 char-5" else lis_chapter[4]

                # 取得'條'
                if class_str == "col-no" and div.find("a"):
                    article = div.find("a").text.replace(' ', '')

                # 取得'項'的內容
                if class_str == "row":
                    int_item = 0
                if class_str == "line-0000":
                    item = div.text
                    lis_chapter_temp = [chapter for chapter in lis_chapter if chapter != '']  # 整合章節內容
                    chapter = ' '.join(lis_chapter_temp)
                    content = f"{law_name} {chapter} {article} {item}"  # 整合法條內容
                    lis_laws_contents.append(content)
                elif class_str == "line-0000 show-number":
                    int_item += 1
                    item_num = f"第{str(int_item)}項"
                    item = f"{item_num} {div.text}"
                    lis_chapter_temp = [chapter for chapter in lis_chapter if chapter != '']  # 整合章節內容
                    chapter = ' '.join(lis_chapter_temp)
                    content = f"{law_name} {chapter} {article} {item}"  # 整合法條內容
                    lis_laws_contents.append(content)

                # 取得'款'的內容
                if class_str == "line-0004" or class_str == "line-0006":
                    paragraph = f"第{div.text.split('、')[0]}款 {div.text.split('、')[-1]}"
                    if int_item != 0:
                        lis_chapter_temp = [chapter for chapter in lis_chapter if chapter != '']  # 整合章節內容
                        chapter = ' '.join(lis_chapter_temp)
                        law = f"{law_name} {chapter} {article} {item_num} {paragraph}"  # 整合法條內容
                        lis_laws_contents.append(law)
                    else:
                        lis_chapter_temp = [chapter for chapter in lis_chapter if chapter != '']  # 整合章節內容
                        chapter = ' '.join(lis_chapter_temp)
                        law = f"{law_name} {chapter} {article} {paragraph}"  # 整合法條內容
                        lis_laws_contents.append(law)

            # 紀錄法條內容
            lis_laws.append(lis_laws_contents)
        else:
            print(f">>> Unable to access url: {url}\n>>> Status code: {response_law.status_code}")
else:
    print(f">>> Unable to access url: {url}\n>>> Status code: {response.status_code}")

# 建立存放資料夾
dir_laws = os.path.join(os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"), "laws")
os.makedirs(dir_laws, exist_ok=True)

# 儲存成文字檔案
fil_laws = os.path.join(dir_laws, "laws_and_content.txt")
with open(fil_laws, "w", encoding="utf-8") as f:
    for lis_law in lis_laws:
        for law in lis_law:
            law = law.strip('\n') + '\n'
            f.write(law)
