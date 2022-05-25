
text = "This is my test text. We're keeping this text short to keep things manageable"

text = text.lower()
skips = [".", ",", ";", ":", "'", '"']
for ch in skips:
    text = text.replace(ch, "") 

def count_word(text):
    """Count the number of times each word occurs in text (str). Return dictionary 
    where keys are unique words and values are word counts.Skip punctuation."""
    
    word_counts = {}
    for word in text.split(" "):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts


#una forma más rápida de hacerlo es con Counter (1 línea de código)
from collections import Counter

def count_word_fast(text):
    """Count the number of times each word occurs in text (str). Return dictionary 
    where keys are unique words and values are word counts.Skip punctuation."""
    
    word_counts = Counter(text.split(" "))
    return word_counts
