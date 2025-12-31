import os
import requests
import argparse

PEXELS_API_KEY = "FVnVWOGne4gtyTiDwjxvq9eJf3FGzO3DxcUwH0859g1aLO0K2yPZ87O2"
PEXELS_API_URL = "https://api.pexels.com/v1/search"

SEARCH_TERMS = {
    "hand_raised": "student raising hand classroom",
    "writing": "student writing notebook",
    "looking_board": "person looking at whiteboard",
    "device_use": "student using phone classroom"
}

FOLDER_PATHS = {
    "hand_raised": "data/train/hand_raised/",
    "writing": "data/train/writing/",
    "looking_board": "data/train/looking_board/",
    "device_use": "data/train/device_use/"
}

def download_images(search_term, output_folder, limit=30):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_term, "per_page": limit}
    response = requests.get(PEXELS_API_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    photos = data.get("photos", [])
    os.makedirs(output_folder, exist_ok=True)
    for idx, photo in enumerate(photos):
        url = photo["src"].get("large")
        if url:
            img_data = requests.get(url).content
            file_path = os.path.join(output_folder, f"img_{idx+1}.jpg")
            with open(file_path, "wb") as f:
                f.write(img_data)
            print(f"Downloaded {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Download images from Pexels API.")
    parser.add_argument("--class", choices=list(SEARCH_TERMS.keys()) + ["all"], required=True, help="Target class to download images for, or 'all' for all classes.")
    parser.add_argument("--limit", type=int, default=100, help="Number of images to download per class.")
    args = parser.parse_args()
    if args.__dict__["class"] == "all":
        for class_name in SEARCH_TERMS.keys():
            print(f"Downloading for class: {class_name}")
            search_term = SEARCH_TERMS[class_name]
            output_folder = FOLDER_PATHS[class_name]
            download_images(search_term, output_folder, args.limit)
    else:
        class_name = args.__dict__["class"]
        search_term = SEARCH_TERMS[class_name]
        output_folder = FOLDER_PATHS[class_name]
        download_images(search_term, output_folder, args.limit)

if __name__ == "__main__":
    main()
