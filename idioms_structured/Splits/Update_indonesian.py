import json

# Define the file paths
indonesian_file = 'idioms_structured/Span_tagged_data/Indonesian/Final_Indonesian_MERGED.jsonl'
test_file = 'idioms_structured/Splits/test.jsonl'

indonesian_examples_formatted = []

# Read and parse Indonesian data
with open(indonesian_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        # Extract common fields, handling Indonesian file's specific key casing
        idiom_id = data.get('idiom_id')
        idiom = data.get('idiom')
        meaning_id = data.get('meaning_id')
        sense_number = data.get('sense_number')
        idiomaticity = data.get('Idiomaticity', '').lower()
        register = data.get('Register', [])
        region = data.get('Region', [])
        
        # Extract individual examples from the 'examples' list
        examples = data.get('examples', [])
        for ex in examples:
            formatted_ex = {
                "language": "Indonesian",
                "idiom_id": idiom_id,
                "idiom": idiom,
                "meaning_id": meaning_id,
                "sense_number": sense_number,
                "idiomaticity": idiomaticity,
                "register": register,
                "region": region,
                "sentence": ex.get('sentence'),
                "span_start": ex.get('span_start'),
                "span_end": ex.get('span_end'),
                "matched_span": ex.get('matched_span')
            }
            indonesian_examples_formatted.append(formatted_ex)

# Append formatted examples to test.jsonl
with open(test_file, 'a', encoding='utf-8') as f:
    for ex in indonesian_examples_formatted:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"Successfully added {len(indonesian_examples_formatted)} Indonesian examples to {test_file}")