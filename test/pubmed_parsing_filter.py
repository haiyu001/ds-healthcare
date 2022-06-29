import json
from pprint import pprint
from utils.general_util import dump_json_file, load_json_file, get_filepaths_recursively, split_filepath
import pandas as pd
import os


input_dir = "/Users/haiyang/pubmed/update_json_files"
output_dir = "/Users/haiyang/pubmed/update_json_files_en"

valid_nums = list(range(1321, 1370))

ids = set()
json_filepaths = get_filepaths_recursively(input_dir, ["json"], sort=True)
for json_filepath in json_filepaths:
    _, file_name, _ = split_filepath(json_filepath)
    file_name = file_name.rsplit('.')[0]

    num = int(file_name[9:])
    if num not in valid_nums:
        print(f"skip parsing {json_filepath}")
        continue
    else:
        print(f"parsing {json_filepath}")

    article_records = []

    with open(json_filepath, "r") as input:
        min_year, max_year = None, None
        for line in input:
            article_record = json.loads(line)
            if article_record["lang"] != "eng":
                continue
            abstract_text = article_record.get("abstract")
            if not abstract_text:
                continue
            else:
                try:
                    if isinstance(abstract_text, str):
                        article_record["abstract"] = abstract_text
                    elif isinstance(abstract_text, dict):
                        if "#text" in abstract_text:
                            article_record["abstract"] = abstract_text["#text"]
                        elif "b" in abstract_text:
                            article_record["abstract"] = abstract_text["b"]
                        elif "i" in abstract_text:
                            article_record["abstract"] = abstract_text["i"]
                        elif "sup" in abstract_text:
                            article_record["abstract"] = abstract_text["sup"]
                        elif "sub" in abstract_text:
                            article_record["abstract"] = abstract_text["sub"]
                        elif "@Label" in abstract_text:
                            continue
                        else:
                            raise Exception(f"bad key in abstract_text")
                    elif isinstance(abstract_text, list):
                        article_record["abstract"] = \
                            "\n".join([section["#text"] for section in abstract_text if section and "#text" in section])
                    else:
                        raise Exception(f"new abstract data format found for "
                                        f"article id [{article_record['article_id']}] in file [{file_name}]")
                except Exception as e:
                    pprint(abstract_text)
                    raise e

            del article_record["title"]
            del article_record["lang"]
            article_records.append(article_record)

            year = article_record["year"]
            if len(article_record["year"]) != 4:
                if " " in year:
                    parts = year.split(" ", 1)
                    if parts[0][:4].isdigit():
                        article_record["year"] = parts[0][:4]
                    elif parts[-1][:4].isdigit():
                        article_record["year"] = parts[-1][:4]
                elif "-" in year:
                    article_record["year"] = year.split("-")[0]
                elif article_record["article_id"] == "32422596":
                    article_record["year"] = "2019"
                if len(article_record["year"]) != 4:
                    print(year, "/", article_record["year"])
                    raise Exception(article_record["article_id"])

            min_year = article_record["year"] if min_year is None else min(min_year, article_record["year"])
            max_year = article_record["year"] if max_year is None else max(max_year, article_record["year"])

    if len(article_records) > 0:
        article_pdf = pd.DataFrame(article_records)
        article_filepath = os.path.join(output_dir, f"{file_name}-{min_year}-{max_year}.json")
        article_pdf.to_json(article_filepath, orient="records", lines=True, force_ascii=False)
    else:
        print(f"no valid articles in {json_filepath}")

input_dir = "/Users/haiyang/pubmed/base_json_files_en"
output_filepath = "/Users/haiyang/pubmed/base_articles.json"
json_filepaths = get_filepaths_recursively(input_dir, ["json"], sort=True)

count = 0
with open(output_filepath, "w") as output:
    for json_filepath in json_filepaths:
        with open(json_filepath, "r") as input:
            for line in input:
                output.write(line)
                count += 1
                if count % 1000000 == 0:
                    print(count)


def load_base(base_file_path):
    base_dict = dict()
    count = 0
    with open(base_file_path, "r") as input:
        for line in input:
            article_record = json.loads(line)
            article_id = article_record["article_id"]
            base_dict[article_id] = line
            count += 1
            if count % 1000000 == 0:
                print(count)
    return base_dict

base_file_path = "/Users/haiyang/pubmed/base_articles.json"
base_dict = load_base(base_file_path)
print(len(base_dict))

valid_nums = list(range(1115, 1370))

update_dir = "/Users/haiyang/pubmed/update_json_files_en"
json_filepaths = get_filepaths_recursively(update_dir, ["json"], sort=True)
for json_filepath in json_filepaths:
    _, file_name, _ = split_filepath(json_filepath)
    file_name = file_name.rsplit('.')[0]

    num = int(file_name[9:13])
    if num not in valid_nums:
        print(f"skip parsing {json_filepath}")
        continue
    else:
        print(f"parsing {json_filepath}")

    with open(json_filepath, "r") as input:
        for line in input:
            article_record = json.loads(line)
            article_id = article_record["article_id"]
            base_dict[article_id] = line

pubmed_articles_filepath = "/Users/haiyang/pubmed/pubmed_articles.json"
count = 0
with open(pubmed_articles_filepath, "w") as output:
    for _, line in base_dict.items():
        output.write(line)
        count += 1
        if count % 1000000 == 0:
            print(count)







