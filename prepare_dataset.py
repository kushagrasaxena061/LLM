import os
import urllib.request
import json
import gzip

print("📥 Step 1/3: Downloading TinyStories (High-quality English & Reasoning)...")
tinystories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
stories_path = "tinystories_valid.txt"

if not os.path.exists(stories_path):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(tinystories_url, stories_path)
    print("✓ Downloaded TinyStories validation/pretraining subset.")
else:
    print("✓ Found existing TinyStories file.")

print("📥 Step 2/3: Generating Python & Technical Knowledge Corpus...")
python_snippets = """
# Python Standard Library & Algorithm Snippets
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        if not self.head:
            self.head = Node(value)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = Node(value)

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

import json
import math
import os
import sys
import time

def calculate_transformer_params(d_model: int, n_layers: int, vocab_size: int):
    embedding_params = vocab_size * d_model
    layer_params = n_layers * (4 * d_model * d_model + 2 * (4 * d_model * d_model))
    return embedding_params + layer_params
"""

print("📦 Step 3/3: Assembling unified 'corpus.txt'...")
with open(stories_path, "r", encoding="utf-8") as f_in, open("corpus.txt", "w", encoding="utf-8") as f_out:
    stories_data = f_in.read()
    f_out.write(stories_data)
    f_out.write("\n\n" + (python_snippets * 50))

file_size_mb = os.path.getsize("corpus.txt") / (1024 * 1024)
print(f"🎉 Successfully created 'corpus.txt'! Total size: {file_size_mb:.2f} MB")
