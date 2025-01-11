import argparse
import json

from tqdm import tqdm

from utils.inference import api_generate, api_embed

def main():
    parser = argparse.ArgumentParser(description='Synthetic generation script.')
    parser.add_argument('--gen_model', type=str, default='gpt-4o-mini', help='Model path for generation')
    parser.add_argument('--emb_model', type=str, default='openai/text-embedding-3-small', help='Model path for embedding')
    args = parser.parse_args()

    with open('./data/emoji-info/characters.txt', 'r', encoding='utf-8') as file:
        emoji_characters = file.read().splitlines()
    emoji_to_index = {emoji: idx for idx, emoji in enumerate(emoji_characters)}

    description_templates = {
        'self': '{emoji}',
        'similar': 'Find emojis similar to the emoji {emoji}. Choose between ten and twenty emojis, return these in a comma-separated list, and return nothing else.',
        'oneword': 'Describe the emoji {emoji} in one word. Write the five best words in a comma-separated list and return nothing else.',
        'sentence': 'Describe the emoji {emoji} in one sentence. Write five different and diverse sentences. Each sentence should contain no commas and the period at the end of the sentence should be replaced by a comma. Return the five sentences in a comma-separated list and return nothing else.',
        'slang': 'Describe the emoji {emoji} in slang. Examples of slang include words that one uses to text, Twitter acronyms and sayings, potential colloquial associations of the emoji, and words relating to culture and memes. Use variety to encourage all kinds of slang are covered. Write ten different slang terms in a comma-separated list and return nothing else.',
    }
    similar_emoji_template = 'Find emojis that match the description \'{description}\'. Choose between ten and twenty emojis, return these in a comma-separated list, and return nothing else.'

    # Create a JSONL file to save script data
    output_file_path = './data/synthetic_generation_output.jsonl'
    with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
        # Save metadata to the start of the file
        metadata = {
            'description': 'Synthetic generation script output',
            'gen_model': args.gen_model,
            'emb_model': args.emb_model,
            'description_templates': description_templates,
            'similar_emoji_template': similar_emoji_template
        }
        jsonl_file.write(json.dumps(metadata) + '\n')

    errors = []
    for emoji in tqdm(emoji_characters):
        try:
            all_descriptions = []
            prompts = []
            desc_type_map = []
            emoji_results = []  # Buffer for current emoji only
            
            for desc_type, template in description_templates.items():
                if desc_type == 'self':
                    all_descriptions.append(emoji)
                    desc_type_map.append(desc_type)
                else:
                    prompt = template.format(emoji=emoji)
                    prompts.append(prompt)
            
            # Batch generate all descriptions
            generated_descriptions_unsplit = api_generate([p for p in prompts], model=args.gen_model)
            desc_lists = [
                [desc.strip() for desc in desc_str.split(',') if desc.strip()]
                for desc_str in generated_descriptions_unsplit
            ]
            
            # Process generated descriptions
            for desc_type, descs in zip(desc_type_map, desc_lists):
                all_descriptions.extend(descs)
                desc_type_map.extend([desc_type] * len(descs))
            
            # Get embeddings for all descriptions at once
            all_embeddings = api_embed(all_descriptions, model=args.emb_model)
            
            # Generate similar emoji prompts for all descriptions
            all_similar_prompts = [similar_emoji_template.format(description=desc) for desc in all_descriptions]
            all_similar_emojis_gen = api_generate(all_similar_prompts, model=args.gen_model)
            
            # Process and save results
            for desc_type, description, embedding, similar_emojis_gen in zip(
                desc_type_map, all_descriptions, all_embeddings, all_similar_emojis_gen
            ):
                similar_emojis = [i.strip() for i in similar_emojis_gen.split(',') if i.strip() in emoji_characters]
                
                # Find the original prompt for this description
                prompt = None
                if desc_type != 'self':
                    prompt = description_templates[desc_type].format(emoji=emoji)
                
                emoji_results.append({
                    'emoji': emoji,
                    'type': desc_type,
                    'prompt': prompt,
                    'description': description,
                    'similar_emojis': similar_emojis,
                    'emoji_indices': [emoji_to_index[emoji]] + [
                        emoji_to_index[e] for e in similar_emojis 
                        if e in emoji_to_index
                    ],
                    'embedding': embedding.tolist()
                })

            # Write all results for current emoji
            with open(output_file_path, 'a', encoding='utf-8') as jsonl_file:
                for result in emoji_results:
                    jsonl_file.write(json.dumps(result) + '\n')
            emoji_results = []  # Clear buffer

        except Exception as e:
            error_info = {
                'emoji': emoji,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            errors.append(error_info)
            continue

    # Update metadata with errors at the end
    with open(output_file_path, 'r+', encoding='utf-8') as jsonl_file:
        # Read existing content
        content = jsonl_file.readlines()
        # Parse and update metadata
        metadata = json.loads(content[0])
        metadata['errors'] = errors
        # Move to start of file and write updated metadata
        jsonl_file.seek(0)
        jsonl_file.write(json.dumps(metadata) + '\n')
        # Write rest of content
        jsonl_file.writelines(content[1:])

if __name__ == '__main__':
    main()