import requests
import re

content = requests.get('https://book.douban.com/').text

pattern = re.compile('<li.*?cover.*?ef="(.*?)".*?le="(.*?)".*?or">(.*?)*?</li>', re.S)
results = re.findall(pattern, content)
for result in results:
    url, name, author = result
    print(url, name, author)