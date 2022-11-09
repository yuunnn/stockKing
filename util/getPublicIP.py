import random
import smtplib
import requests
from email.mime.text import MIMEText
from email.header import Header
import re
import time


def requests_headers():
    """
    Random UA  for every request && Use cookie to scan
    """
    user_agent = [
        'Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.8.1) Gecko/20061010 Firefox/2.0',
        'Mozilla/5.0 (Windows; U; Windows NT 5.0; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.6 '
        'Safari/532.0',
        'Mozilla/5.0 (Windows; U; Windows NT 5.1 ; x64; en-US; rv:1.9.1b2pre) Gecko/20081026 Firefox/3.1b2pre',
        'Opera/10.60 (Windows NT 5.1; U; zh-cn) Presto/2.6.30 Version/10.60',
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; ; rv:1.9.0.14) Gecko/2009082707 Firefox/3.0.14',
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
        'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr; rv:1.9.2.4) Gecko/20100523 Firefox/3.6.4 ( .NET CLR 3.5.30729)',
        'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr-FR) AppleWebKit/533.18.1 (KHTML, like Gecko) Version/5.0.2 '
        'Safari/533.18.5',
        'Mozilla/5.0 (compatible; Bytespider; https://zhanzhang.toutiao.com/) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/70.0.0.0 Safari/537.36']
    UA = random.choice(user_agent)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'User-Agent': UA, 'Upgrade-Insecure-Requests': '1', 'Connection': 'keep-alive', 'Cache-Control': 'max-age=0',
        'Accept-Encoding': 'gzip, deflate, sdch', 'Accept-Language': 'zh-CN,zh;q=0.8',
        "Referer": "http://www.baidu.com/link?url=www.so.com&url=www.soso.com&&url=www.sogou.com"}
    return headers


def get_public_IP():
    r = requests.get('http://txt.go.sohu.com/ip/soip', headers=requests_headers())
    ip = re.findall(r'\d+.\d+.\d+.\d+', r.text)[0]
    return ip


def send_IP(ip):
    mailhost = 'smtp.qq.com'
    qqmail = smtplib.SMTP()
    qqmail.connect(mailhost, 25)
    account = '228353960@qq.com'

    password = 'obuadshusqiacaci'

    qqmail.login(account, password)

    receiver = 'wooxy610@icloud.com'

    content = ip
    message = MIMEText(content, 'plain', 'utf-8')
    subject = 'IP'
    message['Subject'] = Header(subject, 'utf-8')

    qqmail.sendmail(account, receiver, message.as_string())
    qqmail.quit()


if __name__ == "__main__":
    ip = get_public_IP()
    send_IP(ip)
    while True:
        time.sleep(300)
        ip_now = get_public_IP()
        if ip != ip_now:
            ip = ip_now
            send_IP(ip)