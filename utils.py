import json

def read_json(rpath: str):
    try:
        with open(rpath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"error: {rpath} not found.")
    except json.JSONDecodeError:
        print("error: not a json file")
    except Exception as e:
        print(f"error: {e}")

def write_jsonl(anns, wpath):
    pass

def add_visilize2screenshot(image, actions):
    # to add visiual to image

    return image