import json
from typing import Optional

def extract_emoji_data(
    element_field: str, 
    output_file_path: Optional[str] = None,
    input_file_path: str = './data/emoji-info/emoji-api-data.json'
) -> None:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)['data']
        if element_field == 'unicodeName': # Remove prefix, eg "unicodeName": "E5.0 flag: England"
            elements = [' '.join(emoji[element_field].split(' ')[1:]) for emoji in data]
        else:
            elements = [emoji[element_field] for emoji in data]
    
    output_file_path = output_file_path or f'./data/emoji-info/{element_field}s.txt'
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(elements, file, ensure_ascii=False, indent=2)