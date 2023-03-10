import difflib
from icecream import ic

words = ['hello', 'Hallo', 'hi', 'house', 'key', 'screen', 'hallo', 'question', 'format']
matches = difflib.get_close_matches('Hello', words)
ic(matches)
