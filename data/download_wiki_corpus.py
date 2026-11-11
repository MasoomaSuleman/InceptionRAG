from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import json
import bz2
import gzip
import xml.etree.ElementTree as ET
import argparse
from beir.datasets.data_loader import GenericDataLoader

DATASETS = ["nq", "hotpotqa", "msmarco"]

def download(url: str, outfile: Path):
    """Download a file with a progress bar."""

    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists():
        print(f"Skipping {outfile.name}")
        return

    headers = {
    "User-Agent": "InceptionRAGDownloader/1.0"
    }

    r = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60,
    )
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0))

    with open(outfile, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=outfile.name,
    ) as pbar:

        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_msmarco(root):
    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
    docs_path = root / "custom_corpus" / "msmarco" / "msmarco-docs.tsv.gz"
    download(url, docs_path)

    wiki_corpus = root / "custom_corpus" / "msmarco" / "wiki_corpus.jsonl"


    total = 3213835      # number of MS MARCO documents

    with gzip.open(docs_path, "rt", encoding="utf-8") as fin, \
         open(wiki_corpus, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total):

            parts = line.rstrip("\n").split("\t", 3)

            if len(parts) != 4:
                continue

            docid, url, title, body = parts

            page = {
                "id": docid,
                "title": title,
                "text": body,
            }

            fout.write(json.dumps(page, ensure_ascii=False))
            fout.write("\n")

    print(f"Saved {wiki_corpus}")



def download_hotpotqa(root):
    url = "https://hotpotqa.s3.amazonaws.com/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
    wiki_dump_path = root / "custom_corpus" / "hotpotqa" / "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
    download(url, wiki_dump_path)

    og_dataset = root / "daetasets" / "hotpotqa"
    corpus, queries, qrels = GenericDataLoader(og_dataset).load(split="test")

    titles = sorted({doc["title"].strip() for doc in corpus.values() if doc["title"].strip()})
    remaining = len(titles)

    found = 0
    wiki_corpus = root / "custom_corpus" / "hotpotqa" / "wiki_corpus.jsonl"

    with tarfile.open(wiki_dump_path, "r:bz2") as tar, \
            open(wiki_corpus, "w", encoding="utf-8") as out:

        members = [m for m in tar.getmembers() if m.isfile()]

        for member in tqdm(members, desc="Processing shards"):

            f = tar.extractfile(member)
            if f is None:
                continue

            for line in f:

                page = json.loads(line)

                title = page["title"]

                if title in titles:

                    out.write(json.dumps(page, ensure_ascii=False))
                    out.write("\n")
                    # page = {
                    #     "title": title,
                    #     "text": text,
                    # }

                    found += 1
                    remaining =-1
                    titles.remove(title)

                    if found % 10000 == 0:
                        print(f"Found {found:,} pages")

                    if remaining == 0:
                        print("Found every requested page")
                        break               
                    

    print(f"\nExtracted {found:,} pages")
    print(f"Saved to {wiki_corpus}")

    

def download_nq(root):
    print("here")
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
    wiki_dump_path = root / "custom_corpus" / "nq" / "enwiki-latest-pages-articles-multistream.xml.bz2s"
    download(url, wiki_dump_path)

    og_dataset = root / "datasets" / "nq"
    corpus, queries, qrels = GenericDataLoader(
        str(og_dataset)
    ).load(split="test")

    titles = {
        doc["title"].strip()
        for doc in corpus.values()
        if doc["title"].strip()
    }
    remaining = len(titles)

    print(f"Searching for {len(titles):,} Wikipedia pages")

    found = 0

    output_path = root / "custom_corpus" / "nq" / "wiki_corpus.jsonl"

    with bz2.open(wiki_dump_path, "rb") as wiki_file, \
            open(output_path, "w", encoding="utf-8") as out:

        context = ET.iterparse(
            wiki_file,
            events=("end",),
        )

        for _, elem in tqdm(context):

            if not elem.tag.endswith("page"):
                continue

            title_elem = elem.find("./{*}title")

            if title_elem is None:
                elem.clear()
                continue

            title = title_elem.text

            if title not in titles:
                elem.clear()
                continue

            revision = elem.find("./{*}revision")

            text = ""

            if revision is not None:

                text_elem = revision.find("./{*}text")

                if text_elem is not None and text_elem.text is not None:
                    text = text_elem.text

            page = {
                "title": title,
                "text": text,
            }

            out.write(json.dumps(page, ensure_ascii=False))
            out.write("\n")

            found += 1

            if found % 1000 == 0:
                print(f"Found {found:,} pages")

            elem.clear()
            titles.remove(title)
            remaining =- 1
            if remaining == 0:
                print("Found every requested page")
                break

    print(f"\nExtracted {found:,} pages")
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare BEIR datasets")
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--data-root", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    root = Path(args.data_root)
    dataset_path = root / "datasets" / args.dataset
    custom_corpus_path = root / "custom_corpus"
    dataset_path.mkdir(parents=True, exist_ok=True)
    custom_corpus_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == "msmarco" or args.all:
        download_msmarco(root)
    if args.dataset == "hotpotqa" or args.all:
        download_hotpotqa(root)
    if args.dataset == "nq" or args.all:
        download_nq(root)


if __name__ == "__main__":
    main()
