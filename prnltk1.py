import re

def simple_hashtag_tokenizer(hashtag):
    # Remove '#' symbol
    hashtag = hashtag.lstrip('#')
    
    # Split based on capital letters (camel case)
    tokens = re.findall(r'[A-Z][a-z]*', hashtag)
    
    return tokens

# Example
hashtag = "#ILovePython"
tokens = simple_hashtag_tokenizer(hashtag)
print(tokens)