from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from utils.general_util import dump_json_file, load_json_file, get_filepaths_recursively, split_filepath
from gzip import GzipFile
import pandas as pd
import xmltodict
import os


def parse_file(input_filepath, output_filepath):
    _, file_name, _ = split_filepath(input_filepath)

    data_dict = xmltodict.parse(GzipFile(input_filepath))

    article_records = []
    articles = data_dict["PubmedArticleSet"]["PubmedArticle"]

    for article in articles:
        article_record = dict()

        # id
        article_id_dicts = article["PubmedData"]["ArticleIdList"]["ArticleId"]
        if isinstance(article_id_dicts, dict):
            article_id_dicts = [article_id_dicts]
        for article_id_dict in article_id_dicts:
            if article_id_dict["@IdType"] == "pubmed":
                article_record["article_id"] = article_id_dict["#text"]

        article_data = article["MedlineCitation"].get("Article")
        if article_data:
            journal_data = article_data["Journal"]

            # pub year
            pub_date = journal_data["JournalIssue"]["PubDate"]
            if "Year" in pub_date:
                article_record["year"] = pub_date["Year"]
            elif "MedlineDate" in pub_date:
                article_record["year"] = pub_date["MedlineDate"]
            else:
                raise Exception(f"No pub date for {article_record['article_id']}")

            # lang
            article_record["lang"] = article_data["Language"]

            # journal
            article_record["journal"] = journal_data["Title"]

            # title
            article_record["title"] = article_data["ArticleTitle"]

            # abstract
            abstract = article_data.get("Abstract")
            if abstract:
                article_record["abstract"] = abstract["AbstractText"]

        article_records.append(article_record)

    article_pdf = pd.DataFrame(article_records)
    article_pdf.to_json(output_filepath, orient="records", lines=True, force_ascii=False)
    return input_filepath


if __name__ == "__main__":

    input_dir = "/Users/haiyang/pubmed/update_gz_files"
    output_dir = "/Users/haiyang/pubmed/update_json_files"

    valid_nums = list(range(1115, 1370))

    xml_gz_filepaths = get_filepaths_recursively(input_dir, ["gz"], sort=True)
    input_filepath_to_output_filepath = {}
    for xml_gz_filepath in xml_gz_filepaths:
        _, file_name, _ = split_filepath(xml_gz_filepath)
        file_name = file_name.rsplit('.')[0]
        num = int(file_name[9:])
        if num not in valid_nums:
            print(f"skip {xml_gz_filepath}")
            continue
        else:
            print(f"keep {xml_gz_filepath}")
            input_filepath_to_output_filepath[xml_gz_filepath] = os.path.join(output_dir, f"{file_name}.json")

    with ProcessPoolExecutor(max_workers=8) as executor:
        to_do_map = {}
        for input_filepath in input_filepath_to_output_filepath:
            output_filepath = input_filepath_to_output_filepath[input_filepath]
            future = executor.submit(parse_file, input_filepath, output_filepath)
            to_do_map[future] = input_filepath

        done_iter = futures.as_completed(to_do_map)
        for future in done_iter:
            status = future.result()
            print(f"{status} parsed")

