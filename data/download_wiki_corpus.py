from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import json
import bz2
import gzip
import csv
import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

DATASETS = ["nq", "hotpotqa"]

def download(url: str, outfile: Path):
    """Download a file with a progress bar."""

    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists():
        print(f"Skipping download of {outfile.name}")
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
                
                
def load_queries(path: Path) -> List[Dict[str, str]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            items = (json.loads(line) for line in f if line.strip())
        else:
            items = json.load(f)

        for item in items:
            qid = item.get("id") or item.get("query_id") or item.get("_id")
            question = item.get("question") or item.get("query") or item.get("text")
            metadata = item.get("metadata", "")

            if qid is not None and question is not None:
                records.append({
                    "id": str(qid).strip(),
                    "question": str(question).strip(),
                    "metadata": metadata
                })

    return records


def load_qrels_by_query(qrels_paths: Iterable[Path]) -> Dict[str, List[str]]:
    """Build query_id -> list of corpus_ids using train/dev/test qrels files.

    BEIR qrels are often plain TSV files with no header and the layout:
        query-id    corpus-id    score
    but some files may include a header row. This parser supports both.
    """
    corpus_ids_by_query: Dict[str, Set[str]] = defaultdict(set)

    for qrels_path in qrels_paths:
        qrels_path = Path(qrels_path)
        if not qrels_path.exists():
            continue

        with open(qrels_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]

        if not rows:
            continue

        # Skip a header row when present.
        header = [cell.strip().lower() for cell in rows[0]]
        if any(h in {"query-id", "query_id", "queryid", "qid", "query id"} for h in header):
            rows = rows[1:]

        for row in rows:
            if len(row) < 2:
                continue

            qid = str(row[0]).strip()
            corpus_id = str(row[1]).strip()
            if not qid or not corpus_id:
                continue
            corpus_ids_by_query[qid].add(corpus_id)

    return {qid: sorted(corpus_ids) for qid, corpus_ids in corpus_ids_by_query.items()}

def annotate_queries_with_corpus_ids(
    root, dataset
) -> List[Dict[str, str]]:
    """Attach qrels-derived corpus_ids to each query record for debugging."""
    queries_file = root / f"samples/{dataset}_queries_100.json"
    qrels_dir = root / f"datasets/{dataset}/qrels"
    queries = load_queries(queries_file)
    
    qrels_paths = [
        qrels_dir / "train.tsv",
        qrels_dir / "dev.tsv",
        qrels_dir / "test.tsv",
    ]
    qrels_mapping = load_qrels_by_query(qrels_paths)

    annotated: List[Dict[str, str]] = []
    for record in queries:
        updated = dict(record)
        qid = str(updated.get("id") or updated.get("query_id") or "").strip()
        updated["corpus_ids"] = qrels_mapping.get(qid, [])
        annotated.append(updated)

    return annotated

def load_corpus(filename):
    docs = {}

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs[obj["_id"]] = obj

    return docs

def get_doc_titles(root, dataset):
    corpus_file = root / f"datasets/{dataset}/corpus.jsonl"
    
    annotated_queries = annotate_queries_with_corpus_ids(root, dataset)
    corpus = load_corpus(corpus_file)

    titles = []

    for item in annotated_queries:
        for doc_id in item["corpus_ids"]:
            if doc_id in corpus:
                titles.append(corpus[doc_id].get("title", ""))


    return titles

def get_doc_ids(annotated_queries):
    ids = set()

    for item in annotated_queries:
        ids.update(item["corpus_ids"])

    return ids

import html
import re
import mwparserfromhell

def wikitext_to_text(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)

    text = code.strip_code(
        normalize=True,
        collapse=True,
        keep_template_params=False,
    )

    text = html.unescape(text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def golden_passages(root, dataset): 
    golden_output = root / "custom_corpus" / dataset / "golden_passages_corpus.jsonl"
    
    if not golden_output.exists():
        print("Finding query golden passages...")
        corpus_file = root / f"datasets/{dataset}/corpus.jsonl"
        corpus = load_corpus(corpus_file)
        annotated_queries = annotate_queries_with_corpus_ids(root, dataset)
        
        goldens = []

        for item in annotated_queries:
            for doc_id in item["corpus_ids"]:
                if doc_id in corpus:
                    passage = corpus[doc_id].get("text", "")
                    title = corpus[doc_id].get("title", "")
                    qid = item["id"]
                    goldens.append({
                        "id": f"golden-{qid}-{doc_id}",
                        "title": f"golden-{qid}-{doc_id}",
                        "chunk": None,
                        "query_id": qid,
                        "doc_type": "golden",
                        "text": passage,
                    })
        
        with open(golden_output, "w", encoding="utf-8") as f:
            for doc in goldens:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        print(f"Golden Passages corpus saved at {golden_output}")
    else:
        print(f"Golden Passages corpus already exists at {golden_output}")
                


def download_msmarco(root):
    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
    docs_path = root / "custom_corpus" / "msmarco" / "msmarco-docs.tsv.gz"
    download(url, docs_path)
    
    golden_passages(root, "msmarco")
    
    wiki_corpus = root / "custom_corpus" / "msmarco" / "wiki_corpus.jsonl"
    if not wiki_corpus.exists():
        annotated_queries = annotate_queries_with_corpus_ids(root, "msmarco")
        doc_ids = get_doc_ids(annotated_queries)
        
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
    else:
        print(f"Wiki corpus already exists at {wiki_corpus}")





def download_nq_hotpot(root, dataset):
    
    golden_passages(root, dataset)

    output_path = root / "custom_corpus" / dataset / "wiki_corpus.jsonl"
    
    if not output_path.exists():
        url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
        wiki_dump_path = root / "custom_corpus" / dataset / "enwiki-latest-pages-articles-multistream.xml.bz2s"
        download(url, wiki_dump_path)
        titles = get_doc_titles(root, dataset)
        
        print(f"Searching for {len(titles):,} Wikipedia pages")

        found = 0
        remaining = len(titles)
        
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
                    "text": wikitext_to_text(text),
                }

                out.write(json.dumps(page, ensure_ascii=False))
                out.write("\n")

                found += 1

                if found % 1000 == 0:
                    print(f"Found {found:,} pages")

                elem.clear()
                titles.remove(title)
                remaining -= 1
                if remaining == 0:
                    print("Found every requested page")
                    break

        print(f"\nExtracted {found:,} pages")
        print(f"Saved to {output_path}")
    
    else:
        print(f"Wiki corpus already exists at {output_path}")
    


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

    if args.dataset == "hotpotqa" or args.all:
        download_nq_hotpot(root, args.dataset)
    if args.dataset == "nq" or args.all:
        download_nq_hotpot(root, args.dataset)


if __name__ == "__main__":
    main()
