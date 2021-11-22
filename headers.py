ID = 0
NAME = 1
DOMAIN = 2
YEAR_FOUNDED = 3
INDUSTRY = 4
SIZE_RANGE = 5
LOCALITY = 6
COUNTRY = 7
LINKEDIN_URL = 8
EMPLOYEE_ESTIMATE = 9
INDEX = -1
TIMEOUT = 5 # SECONDS
MAX_LEN = 10000 # MAX BYTES
NUM_WORKERS = 8 
LEN_CUTOFF = 500 # MIN BYTES
MAX_SUBPAGES = 35
MIN_EMPLOYEES = 10
NUM_SUBWORKERS = 32
CLASS_BALANCE_THRESH = 50000 # ENQUEUE X OF EACH CLASS
ENTRYPOINT_POSITIVES = {'sitemap', 'site-map'}
TARGET_URLS = "(\.com$)|(\.ai$)|(\.org$)|(\.gov$)|(\.edu$)|(\.co$)|(\.org$)|(\.net$)|(\.mil$)"
DIRECT_SUB_DOMAINS = "(^https://www.(.*)\.com/(.*))|\
                       (^https://www.(.*)\.ai/(.*))|\
                       (^https://www.(.*)\.org/(.*))|\
                       (^https://www.(.*)\.gov/(.*))|\
                       (^https://www.(.*)\.edu/(.*))|\
                       (^https://www.(.*)\.co/(.*))|\
                       (^https://www.(.*)\.org/(.*))|\
                       (^https://www.(.*)\.net/(.*))|\
                       (^https://www.(.*)\.mil/(.*))"
CUSTOM_STOPWORDS = [
    'https', 'http', 'www', 'html', 'navigate', 
    'index', 'us', 'none', 'home', 'careers', 'jobsearch',
    'perspectives', 'investor', 'business', 'relations', 
    'javascript', 'youtube', 'facebook', 'instagram', 'twitter',
    'site', 'search', 'article', 'categories', 'rss', 'explore', 
    'video', 'src', 'hashtag', 'content', 'linkedin', 'company',
    'login', 'news', 'web', 'hash', 'newsroom', 'id', 'join', 'diversity',
    'inclusion', 'signup', 'presentation', 'newsletter', 'corporate',
    'global', 'statements', 'contact', 'press', 'guidelines',
    'worldwide', 'governance', 'directors', 'executive', 'president',
    'cookie', 'pdf', 'jpeg', 'jpg', 'png', 'txt', 'story', 'stories',
    'locations', 'terms', 'privacy', 'responsible', 'responsibility',
    'disclaimer', 'career', 'sustainability', 'product', 'xhtml', 'org',
    'eng', 'lang', 'null', 'blog', 'prev', 'corp', 'sitemap', 'internship',
    'newsfeed', 'ical', 'portal'
]
HEADERS = [
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        'Accept': 'test/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    },
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246',
        'Accept': 'test/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    },
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'Accept': 'test/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    },
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9',
        'Accept': 'test/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    },
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36',
        'Accept': 'test/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    },
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1',
        'Accept': 'test/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    }
]