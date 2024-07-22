# script to replace a certain string like 'rpg_box01' with another like 'rpg_tree'

import os, sys

def replace_string_in_file(file_path, target_string, replacement_string):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    new_content = content.replace(target_string, replacement_string)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

def iterate_and_replace(directory, target_string, replacement_string):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.yaml', '.yml', '.csv', '.txt')):
                file_path = os.path.join(root, file)
                replace_string_in_file(file_path, target_string, replacement_string)
                print(f"Processed {file_path}")

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python exchange_string_in_dir.py <directory> <target_string> <replacement_string>")
        sys.exit(1)

    directory = sys.argv[1]
    target_string = sys.argv[2]
    replacement_string = sys.argv[3]

    iterate_and_replace(directory, target_string, replacement_string)
    print("String replacement completed.")
