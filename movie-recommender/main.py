import ast
import sys
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def resolve_data_file(filename):
    # Support running from any working directory.
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / filename,
        base_dir / "Dataset" / filename,
        base_dir.parent / "Dataset" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Checked: {', '.join(str(p) for p in candidates)}"
    )


def convert(text):
    names = []
    for item in ast.literal_eval(text):
        names.append(item["name"])
    return names


def convert_cast(text):
    names = []
    for item in ast.literal_eval(text)[:3]:
        names.append(item["name"])
    return names


def fetch_director(text):
    for item in ast.literal_eval(text):
        if item["job"] == "Director":
            return [item["name"]]
    return []


def collapse(items):
    return [item.replace(" ", "") for item in items]


def build_recommender_data():
    movies = pd.read_csv(resolve_data_file("movies.csv"))
    credits = pd.read_csv(resolve_data_file("credits.csv"))

    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda text: text.split())

    movies["genres"] = movies["genres"].apply(collapse)
    movies["keywords"] = movies["keywords"].apply(collapse)
    movies["cast"] = movies["cast"].apply(collapse)
    movies["crew"] = movies["crew"].apply(collapse)

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda tokens: " ".join(tokens))
    new_df["title_key"] = new_df["title"].str.lower().str.strip()

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity


def recommend(movie, new_df, similarity):
    query = movie.lower().strip()

    if query not in new_df["title_key"].values:
        print("Movie not found in database.")
        suggestions = get_close_matches(query, new_df["title_key"].tolist(), n=5, cutoff=0.6)
        if suggestions:
            print("\nDid you mean:")
            for title_key in suggestions:
                matched = new_df.loc[new_df["title_key"] == title_key, "title"].iloc[0]
                print(matched)
        return

    movie_index = new_df[new_df["title_key"] == query].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda item: item[1],
    )[1:6]

    print("\nRecommended Movies:\n")
    for item in movies_list:
        print(new_df.iloc[item[0]].title)


def main():
    # Show status early so users know why prompt may take time.
    print("Loading movie recommender, please wait...", flush=True)
    new_df, similarity = build_recommender_data()

    if len(sys.argv) > 1:
        movie_name = " ".join(sys.argv[1:])
    else:
        movie_name = input("Enter a movie name: ")

    recommend(movie_name, new_df, similarity)


if __name__ == "__main__":
    main()
