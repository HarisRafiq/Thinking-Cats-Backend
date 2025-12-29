import re

def split_story_into_sentences(story_text):
    # Split by period, exclamation, or question mark followed by space or newline
    # This regex looks for punctuation that ends a sentence, optionally followed by quote, then whitespace
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

test_stories = [
    """Here is a story about a cat. The cat liked to code. One day, it found a bug! The bug was actually a feature. Everyone rejoiced. The end.""",
    
    """First sentence. Second sentence? Third sentence! Fourth one. Fifth is here. Sixth one finishes it.""",
    
    """This is a tricky one... with ellipses. And "quotes". Does it work? Yes it does. Let's see. Done.""",

    """1. Step one. 2. Step two. 3. Step three. 4. Step four. 5. Step five. 6. Step six."""
]

print("=== Testing Sentence Splitting ===\n")

for i, story in enumerate(test_stories):
    print(f"--- Story {i+1} ---")
    sentences = split_story_into_sentences(story)
    print(f"Original length: {len(story)} chars")
    print(f"Sentence count: {len(sentences)}")
    for j, s in enumerate(sentences):
        print(f"  {j+1}: {s}")
    
    if len(sentences) != 6:
        print(f"WARNING: Expected 6 sentences, got {len(sentences)}")
    print()
