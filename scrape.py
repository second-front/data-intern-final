import csv
import json
import requests
import datetime
import regex as re
import numpy as np
import headers as h
import concurrent.futures
from bs4 import BeautifulSoup
from pymongo import MongoClient
DB_CONNECTION = 'mongodb://localhost:27017'

def url_match(data, link) -> str:
    if len(link.attrs['href']) == 0: return ''
    match = re.search(h.DIRECT_SUB_DOMAINS, link.attrs['href'])
    new_target = ""
    sep = ''
    if match:
        new_target = link.attrs['href']
    elif "." not in link.attrs['href'] and is_entrypoint(link.attrs['href']):
        print("RELATIVE MATCH (SITEMAP): ", link.attrs['href'])
        if link.attrs['href'][0] != '/': sep = '/'
        new_target = data[h.DOMAIN] + sep + link.attrs['href']
    elif "." not in link.attrs['href'] and is_entrypoint(data[h.DOMAIN]):
        print("RELATIVE MATCH (INTRA-SITEMAP): ", link.attrs['href'])
        base_match = re.search("^([^\/]+)\/", data[h.DOMAIN])
        if base_match != None:
            base = base_match.group(1)
        if link.attrs['href'][0] != '/': sep = '/'
        new_target = base + sep + link.attrs['href']
    elif "." not in link.attrs['href']:
        print("RELATIVE MATCH (NON-SITEMAP): ", link.attrs['href'])
        if link.attrs['href'][0] != '/': sep = '/'
        new_target = data[h.DOMAIN] + sep + link.attrs['href']
    return new_target

def find_about_pages(links, data):
    href_list = []
    filtered_links = []
    for link in links: 
        if 'href' in link.attrs.keys():
            char_set = set(link.attrs['href'])
            if char_set.isdisjoint({'#','(',')','@'}) and '.pdf' not in link.attrs['href']:
                href_list.append(link.attrs['href'])
                filtered_links.append(link)
    entry = {
        'id': data[h.ID],
        'origin': complete_url(data[h.DOMAIN]),
        'all_link_count': len(links),
        'links': str(href_list),
    }
    client = MongoClient(DB_CONNECTION)
    client.data.sublinks.insert_one(entry)
    client.close()

    selected_urls = None
    if len(filtered_links) > h.MAX_SUBPAGES:
        selected_urls = []
        indices = np.random.choice(len(filtered_links), h.MAX_SUBPAGES, replace=False)
        for index in indices:
            res = url_match(data, filtered_links[index])
            if res != "" and re.search(h.DIRECT_SUB_DOMAINS, complete_url(res)):
                selected_urls.append(filtered_links[index])
    else: selected_urls = filtered_links

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {
            executor.submit(fetch_html, data, url_match(data, link)): \
                link for link in selected_urls
        }

def extract_text(tags):
    texts = []
    for t in tags:
        inner_match = re.search("<.*>([\w -,.!?\"\']+)<.*>", str(t))
        if inner_match:
            text = inner_match.group(1)
            texts.append(text.replace("\"", '\''))
    text = ' '.join(texts)
    if len(text) < h.LEN_CUTOFF: return None
    else: return text

def complete_url(base_url) -> str:
    if 'https://www.' not in base_url: 
        return 'https://www.' + base_url
    else: return base_url

def is_entrypoint(href) -> bool:
    for elem in h.ENTRYPOINT_POSITIVES:
        if elem in href:
            return True

def find_site_map(links, data) -> list:
    site_map_targets = []
    for link in links:
        if 'href' in link.attrs.keys() and is_entrypoint(link.attrs['href']):
            new_target = url_match(data, link)
            if new_target != "":
                site_map_targets.append(new_target)
    return site_map_targets

def fetch_html(data, base_url):
    error = ''
    if base_url == '': return None
    full_url = complete_url(base_url)
    try:
        result = requests.get(full_url, timeout=h.TIMEOUT, headers=np.random.choice(h.HEADERS))
        status = result.status_code
        print("URL: ", full_url)
        print("STATUS: ", status)
        src = result.content
        soup = BeautifulSoup(src, 'html.parser')
        html_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = extract_text(html_tags)

        if result.status_code != 200: text = u''
        elif result.status_code == 200:
            document = {
                'id': data[h.ID],
                'endpoint': full_url,
                'status': result.status_code,
                'time': datetime.datetime.now(),
                'text': text
            }
            client = MongoClient(DB_CONNECTION)
            if not client.data.companies.find_one({'id': data[h.ID]}):
                company = {
                    'id': data[h.ID],
                    'name': data[h.NAME],
                    'domain': data[h.DOMAIN],
                    'year_founded': data[h.YEAR_FOUNDED],
                    'industry': data[h.INDUSTRY],
                    'size_range': data[h.SIZE_RANGE],
                    'locality': data[h.LOCALITY],
                    'country': data[h.COUNTRY],
                    'linked_in_url': data[h.LINKEDIN_URL],
                    'relevant': data[-1]
                }
                client.data.companies.insert_one(company)
            client.data.documents.insert_one(document)
            client.close()
        return soup
    except requests.exceptions.Timeout:
        error = 'timeout'
    except requests.exceptions.SSLError:
        error = 'too many retries'
    except requests.exceptions.TooManyRedirects:
        error = 'too many redirects'
    except requests.exceptions.ConnectionError:
        error = 'refused connection'
    client = MongoClient(DB_CONNECTION)
    failure = {
        'id': data[h.ID],
        'endpoint': full_url,
        'time': datetime.datetime.now(),
        'error': error
    }
    client.data.failures.insert_one(failure)
    return None

def work_unit(data, is_site_map):
    if is_site_map:
        soup = fetch_html(data, data[h.DOMAIN])
        if soup != None:
            links = soup.find_all("a")
            find_about_pages(links, data)
    else:
        soup = fetch_html(data, data[h.DOMAIN])
        if soup != None:
            links = soup.find_all("a")
            site_map_targets = find_site_map(links, data)
            if len(site_map_targets) == 0:
                find_about_pages(links, data)
            else:
                print(site_map_targets)
                for target in site_map_targets:
                    data[h.DOMAIN] = target
                    work_unit(data, True)

def execute_work(data):
    print('in execute_work...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_url = {
            executor.submit(work_unit, obj, False): \
                obj for obj in data
        }

def load_sam_entities_data(num_enqueues, src_file):
    data = []
    counter = 0
    with open(src_file) as jsonfile:
        entities = json.load(jsonfile)
        for entry in entities['domain_agent']:
            if re.search(h.TARGET_URLS, entry['domain_agent_url']):
                counter += 1
                row = [str(counter), entry['domain_agent_name'], \
                    entry['domain_agent_url'], '', entry['attribute_agent'], \
                        '', '', '', '', '', '']
                data.append(row)
                if counter > num_enqueues: break
    return data

def webscrape(num_enqueues, src_file):
    data = load_sam_entities_data(num_enqueues=num_enqueues, src_file=src_file)
    print(len(data))
    execute_work(data)

"""if __name__ == '__main__':
    webscrape()"""