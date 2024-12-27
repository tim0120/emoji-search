import json
import os
from typing import List

from tqdm import tqdm

from utils.inference import api_generate

def generate_alternates(emoji_character: str, original_description: str, num_alternates: int, num_retries: int = 5) -> List[str]:
    prompt = f"""Generate {num_alternates} alternate descriptions for the emoji character "{emoji_character}" ({original_description}). Try to have each description cover a different possible facet of the emoji character "{emoji_character}". Ensure that the descriptions are not redundant with the original description or one another. Return the list of alternate descriptions in parseable JSON format."""
    
    # retries in case of JSON parsing errors due to API response formatting
    for _ in range(num_retries):
        alternates_json = api_generate([prompt], model="gpt-4o-mini")[0]
        # stripping irrelevant characters from gpt-4o-mini response specifically, other models may have other quirks
        alternates_json = alternates_json.strip("`").strip("json").strip("\n")
        try:
            alternates = json.loads(alternates_json)
            if isinstance(alternates[0], dict):
                alternates = [a['description'] for a in alternates]
            return alternates
        except Exception:
            continue
    
    raise ValueError("Failed to parse JSON after multiple retries.")

def save_alternates(num_alternates: int, batch_size: int = 32) -> None:
    """
    Save alternate emoji descriptions to a JSON file incrementally with batch processing.
    
    Args:
        num_alternates (int): Number of alternate descriptions to generate per emoji
        batch_size (int): Number of emojis to process in each batch before saving
    """
    alternates_path = './data/emoji-info/alternate-descriptions.json'
    progress_path = './data/emoji-info/processing_progress.json'
    os.makedirs(os.path.dirname(alternates_path), exist_ok=True)
    
    # Load or initialize processing progress
    processed_emojis = set()
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as file:
                processed_emojis = set(json.load(file))
        except json.JSONDecodeError:
            processed_emojis = set()
    
    # Load existing alternate descriptions
    existing_data = []
    if os.path.exists(alternates_path):
        try:
            with open(alternates_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        except json.JSONDecodeError:
            existing_data = []
    
    # Create lookup for existing data
    existing_lookup = {item['emoji_character']: item for item in existing_data}
    
    # Read and process emoji data
    with open('./data/emoji-api-data.json', 'r', encoding='utf-8') as file:
        emoji_data = json.load(file)['data']
        
        current_batch = []
        for emoji in tqdm(emoji_data, desc="Processing emojis"):
            emoji_character = emoji['character']
            
            # Skip if already processed
            if emoji_character in processed_emojis:
                continue
                
            original_description = emoji['unicodeName']
            alternates = generate_alternates(
                emoji_character, 
                original_description, 
                num_alternates=num_alternates
            )
            
            new_entry = {
                'emoji_character': emoji_character,
                'input': original_description,
                'alternates': alternates
            }
            
            # Update or add new entry
            existing_lookup[emoji_character] = new_entry
            processed_emojis.add(emoji_character)
            current_batch.append(emoji_character)
            
            # Save after each batch
            if len(current_batch) >= batch_size:
                _save_progress(existing_lookup, processed_emojis, alternates_path, progress_path)
                current_batch = []
                
        # Save any remaining items
        if current_batch:
            _save_progress(existing_lookup, processed_emojis, alternates_path, progress_path)

def _save_progress(existing_lookup: dict, 
                  processed_emojis: set, 
                  alternates_path: str, 
                  progress_path: str) -> None:
    """
    Helper function to save current progress and processed emojis.
    
    Args:
        existing_lookup (dict): Dictionary of processed emoji data
        processed_emojis (set): Set of processed emoji characters
        alternates_path (str): Path to save alternate descriptions
        progress_path (str): Path to save progress tracking
    """
    # Save current data
    with open(alternates_path, 'w', encoding='utf-8') as file:
        json.dump(list(existing_lookup.values()), file, ensure_ascii=False, indent=4)
    
    # Save progress
    with open(progress_path, 'w', encoding='utf-8') as file:
        json.dump(list(processed_emojis), file, ensure_ascii=False)
