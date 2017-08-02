import os
_script_dir = os.path.dirname(os.path.realpath(__file__))

def parse():
    from bs4 import BeautifulSoup
    html = open(os.path.join(_script_dir, 'freq.html'), 'rb')
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('pre')
    rows = [x.split('\t') for x in table.contents if len(x) > 0]
    columns = [x.text.split(':')[1].split(';')[0].split('.')[0].strip() for x in soup.find('ul').find_all('li')]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.columns = columns
    return df

if __name__ == '__main__':
    df = parse()
    df.to_csv('freq.csv')
