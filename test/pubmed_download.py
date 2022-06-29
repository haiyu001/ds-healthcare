import requests
import gzip
import shutil
import os
import time

start = 1115
end = 1369
output_dir = "/Users/haiyang/pubmed/gz_files_update"
pubmed_url_base = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles"

for i in range(start, end + 1):
    filename = f"pubmed22n{str(i).zfill(4)}.xml.gz"
    url = f"{pubmed_url_base}/{filename}"
    print(f"downloading {url}")
    response = requests.request(method="get", url=url)
    with open(os.path.join(output_dir, filename), "wb") as result:
        result.write(response.content)
    time.sleep(5)
