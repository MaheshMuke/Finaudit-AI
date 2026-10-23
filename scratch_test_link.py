import requests, re
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
url = 'https://news.google.com/rss/articles/CBMitwFBVV95cUxNVDJNczB5ZFItMnlnb0RlSEc1OGtuUXFwY3djbEJWZTVpSzhUNDlnYVlQU2VuZmdrbUppVFBYZGQ2bnVLWWdsZTdqamxScG45S3BBLUhsZUtJMVl3VjZERlI5LUstbXM2bnBnaXVBWWo1LXhiUC12bEpMdmRTN1hOLTJ0bWotdDdoR05PUEVMNk9DVE1TX1FzS0FiYl9zZEFDZGNlbElwTmhBVXEzRkVUMnNXb3lzVEU?oc=5'
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.content, 'html.parser')
for a in soup.find_all('a'):
    h = a.get('href')
    if h and 'google' not in h:
        print('FOUND EXT LINK:', h)
