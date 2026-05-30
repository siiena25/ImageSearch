#!/usr/bin/env python3
"""
Build catalog.jsonl for the Android MVP from an ASOS dataset file.

Supported inputs:
1) smartcat/asos-data-embedded parquet
   Expected columns: id, brand, gender, product_type, image_url, title, price
2) UniqueData/asos-e-commerce-dataset csv
   Expected columns: sku, name, category, price, description, images

Usage examples:
  python build_catalog_from_asos.py --input data_with_embeddings.parquet --output catalog.jsonl --limit 1000
  python build_catalog_from_asos.py --input products_asos.csv --output catalog.jsonl --limit 1000
"""
import argparse
import ast
import json
import math
from pathlib import Path

import pandas as pd


def parse_brand_from_description(desc):
    if pd.isna(desc):
        return None
    text = str(desc)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "Brand" in item:
                    return str(item["Brand"]).strip()
    except Exception:
        pass
    return None


def parse_first_image(images_value):
    if pd.isna(images_value):
        return None
    text = str(images_value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    return None


def is_valid_price(value):
    try:
        v = float(value)
        return math.isfinite(v)
    except Exception:
        return False


def load_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def convert_smartcat(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"id", "brand", "gender", "product_type", "image_url", "title", "price"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for smartcat parquet: {sorted(missing)}")

    out = df.copy()
    out = out[(out["gender"].astype(str).str.lower() == "women") &
              (out["product_type"].astype(str).str.lower() == "dresses")]
    out = out[["id", "title", "image_url", "brand", "price"]].copy()
    out = out.rename(columns={
        "id": "product_id",
        "image_url": "image"
    })
    return out


def convert_uniquedata(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"sku", "name", "category", "price", "description", "images"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for UniqueData csv: {sorted(missing)}")

    out = df.copy()
    cat = out["category"].astype(str).str.lower()
    name = out["name"].astype(str).str.lower()
    out = out[cat.str.contains("dress", na=False) | name.str.contains("dress", na=False)].copy()

    out["brand"] = out["description"].apply(parse_brand_from_description)
    out["image"] = out["images"].apply(parse_first_image)

    out = out.rename(columns={
        "sku": "product_id",
        "name": "title",
    })[["product_id", "title", "image", "brand", "price"]]

    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_id"] = df["product_id"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["image"] = df["image"].astype(str).str.strip()
    df["brand"] = df["brand"].fillna("Unknown").astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df[df["image"].notna() & (df["image"] != "None") & (df["image"] != "")]
    df = df[df["title"].notna() & (df["title"] != "")]
    df = df[df["product_id"].notna() & (df["product_id"] != "")]
    df = df[df["price"].apply(is_valid_price)]

    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input csv/parquet file")
    parser.add_argument("--output", default="catalog.jsonl", help="Output jsonl path")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of items")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_df(input_path)

    columns = set(df.columns)
    if {"id", "brand", "gender", "product_type", "image_url", "title", "price"} <= columns:
        catalog = convert_smartcat(df)
    elif {"sku", "name", "category", "price", "description", "images"} <= columns:
        catalog = convert_uniquedata(df)
    else:
        raise ValueError(
            "Unknown dataset schema. Expected either smartcat parquet or UniqueData csv schema."
        )

    catalog = normalize(catalog)
    catalog = catalog.head(args.limit)

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in catalog.iterrows():
            obj = {
                "product_id": row["product_id"],
                "title": row["title"],
                "image": row["image"],
                "brand": row["brand"],
                "price": float(row["price"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(catalog)} items to {output_path}")


if __name__ == "__main__":
    main()
