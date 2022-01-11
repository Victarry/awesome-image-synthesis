import re
from semanticscholar import SemanticScholar
sch = SemanticScholar(timeout=2)

p = re.compile('https://arxiv.org/abs/\d+\.\d+')

lines = []
with open('./README.md', 'r') as f:
    for line in f.readlines():
        m = p.search(line)
        if m:
            arxiv_url = m.group()
            try:
                paper = sch.paper(f'URL:{arxiv_url}')
            except Exception:
                print(f"Unsuccessfuly when handle {arxiv_url}")
            else:
                if "numCitedBy" in paper:
                    cite_count = paper["numCitedBy"]
                    if "Cited" in line:
                        line = line.split(" Cited")[0]
                    line = line.strip()
                    line = f'{line} Cited:`{cite_count}`\n'
        lines.append(line)

with open('./README.md', 'w') as f:
    for line in lines:
        f.write(line)