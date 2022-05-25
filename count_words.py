
title_path = './books/Romeo and Juliet.txt' #Plain text utf8 - Gutenberg Project
    
def read_book(title_path):
    """Read a book and return it as a string."""
    with open(title_path, 'r', encoding="utf8") as current_file:
        text = current_file.read()
        text =text.replace ("\n", "").replace("\r", "")
        text = text.lower()
        skips = [".", ",", ";", ":", "'", '"']
        for ch in skips:
            text = text.replace(ch, "") 
    return text    

from collections import Counter

def count_word_fast(text):
    """Count the number of times each word occurs in text (str). Return dictionary 
    where keys are unique words and values are word counts.Skip punctuation."""
    
    word_counts = Counter(text.split(" "))
    return word_counts

def word_stats(word_counts):
    """Return number of unique words and word frequencies. """
    num_unique = len (word_counts)
    counts = word_counts.values()
    return (num_unique, counts)

text = read_book(title_path)
word_counts = count_word_fast(text)
(num_unique, counts) = word_stats(word_counts)
num_unique #cantidad de palabras únicas (sin repetir)
sum(counts) #cantidad de palabras totales


