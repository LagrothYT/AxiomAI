import json
import os
import re
import heapq
from collections import Counter, defaultdict
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.id_to_token = {}  # id -> token
        self.merges = {}  # (p1, p2) -> merged_token
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "[HUMAN]", "[GPT]"]
        
    def train(self, texts):
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
            
        # Initial character-level tokens from corpus
        # word_data: word_str -> list of symbols
        # word_freqs: word_str -> frequency
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[" ".join(list(word)) + " </w>"] += 1
        
        # Initial symbols and pair statistics
        parts = {word: word.split() for word in word_freqs}
        stats = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = parts[word]
            for i in range(len(symbols) - 1):
                stats[symbols[i], symbols[i+1]] += freq
        
        # Build initial character vocab
        chars = set()
        for word in word_freqs:
            for char in parts[word]:
                chars.add(char)
        
        start_idx = len(self.vocab)
        for i, char in enumerate(sorted(list(chars))):
            self.vocab[char] = i + start_idx
            self.id_to_token[i + start_idx] = char
            
        # Optimization: Map pair to list of words containing it
        pair_to_words = defaultdict(set)
        for word, symbols in parts.items():
            for i in range(len(symbols) - 1):
                pair_to_words[symbols[i], symbols[i+1]].add(word)
                
        # Merge loop
        num_merges = self.vocab_size - len(self.vocab)
        loop = tqdm(range(num_merges), desc="Training BPE")
        for i in loop:
            if not stats:
                break
            
            # Find best pair by frequency
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] <= 0:
                break
                
            new_id = len(self.vocab)
            new_token = "".join(best_pair)
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges[best_pair] = i
            
            # Efficiently update affected words
            affected_words = list(pair_to_words[best_pair])
            for word in affected_words:
                freq = word_freqs[word]
                symbols = parts[word]
                
                # Find all occurrences of best_pair in this word
                j = 0
                new_symbols = []
                while j < len(symbols):
                    if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
                        # Before merging, remove broken pairs from stats
                        if j > 0:
                            left_pair = (symbols[j-1], symbols[j])
                            stats[left_pair] -= freq
                            pair_to_words[left_pair].discard(word)
                        if j < len(symbols) - 2:
                            # Avoid double counting if the NEXT pair is also a best_pair
                            # But standard BPE merges one at a time.
                            # However, we must remove the right pair because symbols[j+1] will be gone.
                            right_pair = (symbols[j+1], symbols[j+2])
                            stats[right_pair] -= freq
                            pair_to_words[right_pair].discard(word)
                        
                        new_symbols.append(new_token)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                
                # Update stats for newly created pairs in this word
                parts[word] = new_symbols
                for k in range(len(new_symbols) - 1):
                    if new_symbols[k] == new_token or new_symbols[k+1] == new_token:
                        pair = (new_symbols[k], new_symbols[k+1])
                        stats[pair] += freq
                        pair_to_words[pair].add(word)
            
            # Remove best_pair from stats
            del stats[best_pair]
            del pair_to_words[best_pair]
            
            if (i + 1) % 100 == 0:
                loop.set_description(f"BPE merge {i+1}/{num_merges}: {best_pair[0]} + {best_pair[1]} -> {new_token}")

    def encode(self, text):
        if not text:
            return []
            
        # Isolate special tokens
        for special in self.special_tokens:
            text = text.replace(special, f" {special} ")
            
        words = text.split()
        encoded_ids = []
        
        for word in words:
            if word in self.special_tokens:
                encoded_ids.append(self.vocab[word])
                continue
                
            # node: [token, prev_idx, next_idx, is_valid]
            word_chars = list(word) + ["</w>"]
            if len(word_chars) < 2:
                for token in word_chars:
                    encoded_ids.append(self.vocab.get(token, self.vocab["<UNK>"]))
                continue

            nodes = []
            for i, char in enumerate(word_chars):
                nodes.append([char, i - 1, i + 1, True])
            nodes[-1][2] = -1

            queue = []
            def add_pair(l_idx, r_idx):
                pair = (nodes[l_idx][0], nodes[r_idx][0])
                if pair in self.merges:
                    heapq.heappush(queue, (self.merges[pair], l_idx, r_idx))

            for i in range(len(nodes) - 1):
                add_pair(i, i + 1)

            while queue:
                rank, l, r = heapq.heappop(queue)
                if not nodes[l][3] or not nodes[r][3] or nodes[l][2] != r:
                    continue

                # Merge r into l
                new_token = nodes[l][0] + nodes[r][0]
                nodes[l][0] = new_token
                nodes[r][3] = False
                
                # Update links
                next_idx = nodes[r][2]
                nodes[l][2] = next_idx
                if next_idx != -1:
                    nodes[next_idx][1] = l

                # Add new potential pairs
                prev_idx = nodes[l][1]
                if prev_idx != -1:
                    add_pair(prev_idx, l)
                if next_idx != -1:
                    add_pair(l, next_idx)

            # Collect tokens
            curr = 0
            while curr != -1:
                if nodes[curr][3]:
                    token = nodes[curr][0]
                    encoded_ids.append(self.vocab.get(token, self.vocab["<UNK>"]))
                curr = nodes[curr][2]
                
        return encoded_ids

    def decode(self, ids, clean_up=False):
        tokens = [self.id_to_token.get((i if isinstance(i, int) else i.item()), "<UNK>") for i in ids]
        text = ""
        for token in tokens:
            if token in self.special_tokens:
                text += f"{token} "
            elif token.endswith("</w>"):
                text += token.replace("</w>", " ")
            else:
                text += token
                
        if clean_up:
            return " ".join(text.split())
        return text

    def save(self, path):
        serializable_merges = {f"{k[0]}<SEP>{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab": self.vocab,
            "merges": serializable_merges,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        obj = cls(vocab_size=data.get("vocab_size", 5000))
        obj.special_tokens = data.get("special_tokens", ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "[HUMAN]", "[GPT]"])
        obj.vocab = data["vocab"]
        obj.vocab_size = len(obj.vocab)
        obj.id_to_token = {int(v): k for k, v in obj.vocab.items()}
        
        obj.merges = {}
        for k, v in data["merges"].items():
            p1, p2 = k.split("<SEP>")
            obj.merges[(p1, p2)] = v
            
        return obj
