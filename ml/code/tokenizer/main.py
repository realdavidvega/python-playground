
import tiktoken


def tokenize_and_print_file(file_path):
    '''
      Receives a file path, tokenizes each line and prints the result.
      Prints the result with the input and tokens separated by a comma.
    '''
    encoding = tiktoken.get_encoding("o200k_base")
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = encoding.encode(line.strip())
            if ',' in line:
                print(f"\"{line.strip()}\",\"{tokens}\"")
            else:
                print(f"{line.strip()},\"{tokens}\""),

file = 'resources/o200k.txt'
tokenize_and_print_file(file)
