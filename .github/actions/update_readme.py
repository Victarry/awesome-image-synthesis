import re
from semanticscholar import SemanticScholar
sch = SemanticScholar(timeout=10)

p = re.compile('https://arxiv.org/abs/\d+\.\d+')

CITATION_KEY = "citationCount"

lines = []
with open('./README.md', 'r') as f:
    for line in f.readlines():
        match = re.search('https://arxiv.org/abs/(\d+\.\d+)', line)
        if match:
            arxiv_url = match.group(1)
            print(f"Match paper id {arxiv_url}")
            try:
                target_url = f'arXiv:{arxiv_url}'
                print(target_url)
                paper = sch.get_paper(target_url, fields=[CITATION_KEY, ])
            except Exception as e:
                print(f"Unsuccessfuly when handle {arxiv_url}")
                print(f"Error is {e}")
            else:
                print(paper)
                if CITATION_KEY in paper.keys():
                    cite_count = paper[CITATION_KEY]
                    if "Cited" in line:
                        line = line.split(" Cited")[0]
                    line = line.strip()
                    line = f'{line} Cited:`{cite_count}`\n'
        lines.append(line)

with open('./README.md', 'w') as f:
    for line in lines:
        f.write(line)
