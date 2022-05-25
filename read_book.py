
title_path = './books/English/Otros/The intermediate sex.txt' #Plain text utf8 - Gutenberg Project
def read_book(title_path):
    """Read a book and return it as a string."""
    with open(title_path, 'r', encoding="utf8") as current_file:
        text = current_file.read()
        text =text.replace ("\n", "").replace("\r", "")
    return text

text = read_book(title_path)
len(text)

ind = text.find("Urning men and women,")
ind

sample_text = text [ind:ind+1000]
sample_text

def word_stats(word_counts):
    """Return number of unique words and word frequencies. """
    num_unique = len (word_counts)
    counts = word_counts.values()
    return (num_unique, counts)