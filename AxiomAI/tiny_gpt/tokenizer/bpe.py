import json
import os
import re
from collections import Counter, defaultdict
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.id_to_token = {}  # id -> token
        self.merges = {}  # (p1, p2) -> merged_token
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "[HUMAN]", "[GPT]"]
        
    def _get_stats(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def _merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_token = ''.join(pair)
        for word in v_in:
            w_out = p.sub(new_token, word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, texts):
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
            
        # Initial character-level vocab from corpus
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[" ".join(list(word)) + " </w>"] += 1
        
        # Build initial character vocab
        chars = set()
        for word in word_freqs:
            for char in word.split():
                chars.add(char)
        
        start_idx = len(self.vocab)
        for i, char in enumerate(sorted(list(chars))):
            self.vocab[char] = i + start_idx
            self.id_to_token[i + start_idx] = char
            
        # Merge loop
        num_merges = self.vocab_size - len(self.vocab)
        loop = tqdm(range(num_merges), desc="Training BPE")
        for i in loop:
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            new_id = len(self.vocab)
            new_token = "".join(best_pair)
            
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges[best_pair] = i # Store merge index as rank
            
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            
            if (i + 1) % 100 == 0:
                loop.set_description(f"BPE merge {i+1}/{num_merges}: {best_pair[0]} + {best_pair[1]} -> {new_token}")

    def encode(self, text):
        if not text:
            return []
            
        # Isolate special tokens so they become distinct "words"
        for special in self.special_tokens:
            text = text.replace(special, f" {special} ")
            
        # Split into words (special tokens are now isolated as their own words)
        words = text.split()
        encoded_ids = []
        
        for word in words:
            if word in self.special_tokens:
                encoded_ids.append(self.vocab[word])
                continue
                
            # Prepare word as list of characters
            word_chars = list(word) + ["</w>"]
            
            while len(word_chars) >= 2:
                # Find the earliest learned merge that applies to any pair in current word
                best_pair = None
                best_rank = float('inf')
                
                for i in range(len(word_chars) - 1):
                    pair = (word_chars[i], word_chars[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                
                if best_pair is None:
                    break
                
                # Apply the merge
                new_word_chars = []
                i = 0
                while i < len(word_chars):
                    if i < len(word_chars) - 1 and (word_chars[i], word_chars[i+1]) == best_pair:
                        new_word_chars.append("".join(best_pair))
                        i += 2
                    else:
                        new_word_chars.append(word_chars[i])
                        i += 1
                word_chars = new_word_chars
                
            for token in word_chars:
                token_id = self.vocab.get(token, self.vocab["<UNK>"])
                encoded_ids.append(token_id)
                
        return encoded_ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        text = ""
        for token in tokens:
            if token in self.special_tokens:
                text += f" {token} "
            elif token.endswith("</w>"):
                text += token.replace("</w>", " ")
            else:
                text += token
        return " ".join(text.split())

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
