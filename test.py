import json
from collections import Counter
from pathlib import Path
p=Path("data/meta_Amazon_Fashion.jsonl")
c=Counter()
with p.open("r", encoding="utf-8") as f:
    for line in f:
        line=line.strip().strip(",")
        if line in ("[","]",",","] ,"): continue
        try:
            o=json.loads(line)
        except: 
            continue
        for k in ["id","asin","ASIN","parent_asin","item_id","product_id","sku"]:
            if k in o and o[k] not in (None,""):
                c[k]+=1
print(c)