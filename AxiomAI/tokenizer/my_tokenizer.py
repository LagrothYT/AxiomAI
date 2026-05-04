import json
import os
import re
from collections import defaultdict
from axiom_ui import clear_screen, print_done, print_run_banner, print_step

class CharTokenizer:
    """Byte-Pair Encoding (BPE) tokenizer with pre-tokenization and special tokens.

    Key improvements over a naive BPE:
    1. Pre-tokenization splits text into word chunks (via regex) BEFORE byte encoding.
       This prevents cross-word merges like merging "e " + "t" across word boundaries.
    2. Frequency dictionary training: instead of scanning the full token array on every
       merge step, we track unique word frequency counts. Merges operate on unique words
       only, weighted by their count. Mathematically identical, 1000x+ faster.
    3. Proper special tokens: <PAD>, <BOS>, <EOS> with reserved IDs.
    """

    # GPT-2 inspired pre-tokenization pattern (pure stdlib re, no external deps).
    # Splits text into: contractions, letter words (with optional leading space),
    # number runs, punctuation runs (with optional leading space), whitespace.
    PRETOKENIZE_RE = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+""",
        re.UNICODE
    )

    def __init__(self):
        self.merges = {}          # (int, int) -> int
        self.vocab = {}           # int -> bytes
        self.merge_list = []      # ordered list of merges for encoding

        # Special token IDs (assigned during training or loading)
        self.pad_id = None
        self.bos_id = None
        self.eos_id = None

    # ------------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------------

    def _pretokenize(self, text):
        """Split text into word chunks respecting linguistic boundaries."""
        return self.PRETOKENIZE_RE.findall(text)

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_text_from_jsonl(self, path):
        text = ""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    for msg in data.get('conversations', []):
                        role = msg.get('from', 'unknown')
                        value = msg.get('value', '')
                        text += f"<{role}>: {value}\n"
                except json.JSONDecodeError:
                    continue
        return text

    def _iter_data_files(self, path):
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                full_path = os.path.join(path, name)
                if os.path.isfile(full_path):
                    yield full_path
        else:
            yield path

    def _extract_text_from_path(self, path):
        text_parts = []
        for file_path in self._iter_data_files(path):
            lower = file_path.lower()
            if lower.endswith('.jsonl'):
                text = self._extract_text_from_jsonl(file_path)
                if text:
                    text_parts.append(text)
            elif lower.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    text_parts.append(text)
            else:
                print(f"  Skipping unsupported tokenizer source: {os.path.basename(file_path)}")
        return "\n".join(text_parts)

    # ------------------------------------------------------------------
    # BPE merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pair_counts(word_freqs):
        """Count adjacent token pairs across all words, weighted by word frequency."""
        counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += freq
        return counts

    @staticmethod
    def _merge_pair_in_word(word, pair, new_id):
        """Replace all occurrences of `pair` in a word tuple with `new_id`."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_file(self, text_path, vocab_size, save_path):
        """Train BPE with pre-tokenization. Final vocab == exactly `vocab_size`."""
        clear_screen()
        print_run_banner("Tokenizer Training", [
            ("🧩 Source", text_path),
            ("📦 Vocab", f"{int(vocab_size):,} target tokens"),
            ("🧷 Special", "<PAD>  <BOS>  <EOS>"),
            ("💾 Save", save_path),
        ])

        print_step("Scanning", "extracting source text")
        if not os.path.exists(text_path):
            print("Error: tokenizer source path not found.")
            return False

        raw_text = self._extract_text_from_path(text_path)
        if not raw_text:
            print("Error: No valid .jsonl or .txt tokenizer data found.")
            return False

        num_special = 3  # <PAD>, <BOS>, <EOS>
        min_vocab = 256 + num_special
        if vocab_size < min_vocab:
            print(f"Error: vocab_size must be >= {min_vocab} (256 bytes + {num_special} special tokens). Got {vocab_size}.")
            return False

        # --- Pre-tokenize into word chunks ---
        print_step("Tokenize", "pre-tokenizing text")
        words = self._pretokenize(raw_text)

        # Build frequency dictionary: word (as byte tuple) -> count
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[tuple(word.encode('utf-8'))] += 1

        print_step("Chunks", f"{len(words):,} word chunks  │  {len(word_freqs):,} unique words")

        # --- Initialize byte-level vocab ---
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.merge_list = []
        next_id = 256

        num_merges = vocab_size - 256 - num_special
        print_step("BPE", f"{num_merges:,} merges to reach vocab size {vocab_size:,}")

        for i in range(num_merges):
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                print(f"\n  Exhausted all pairs at merge {i + 1}.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            # Apply merge across all unique words
            new_word_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                merged = self._merge_pair_in_word(word, best_pair, next_id)
                new_word_freqs[merged] += freq
            word_freqs = new_word_freqs

            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = next_id
            next_id += 1

            if (i + 1) % 500 == 0 or i == 0:
                pct = (i + 1) / num_merges * 100
                print(f"\r  {'Merges':<12} {i + 1:,}/{num_merges:,} ({pct:.1f}%)", end="", flush=True)

        print()  # Flush progress line

        # Build ordered merge list (encoding must apply merges in training order)
        self.merge_list = sorted(self.merges.items(), key=lambda x: x[1])

        # --- Append special tokens ---
        self.pad_id = next_id
        self.vocab[self.pad_id] = b'<PAD>'
        next_id += 1

        self.bos_id = next_id
        self.vocab[self.bos_id] = b'<BOS>'
        next_id += 1

        self.eos_id = next_id
        self.vocab[self.eos_id] = b'<EOS>'
        next_id += 1

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.save(save_path)
        print_done([
            ("✅ Done", f"Tokenizer saved to {save_path}"),
            ("📦 Final", f"{len(self.vocab):,} tokens"),
        ])
        return True

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text):
        """Encode text to token IDs with pre-tokenization boundaries enforced."""
        chunks = self._pretokenize(text)
        token_ids = []
        for chunk in chunks:
            ids = list(chunk.encode('utf-8'))
            
            # Smart Search: Iteratively find the highest-priority (lowest rank) merge locally
            while len(ids) > 1:
                best_pair = None
                best_rank = float('inf')
                
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    if pair in self.merges:
                        # Lower token ID == created earlier == higher priority
                        rank = self.merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                
                if best_pair is None:
                    break  # No more mathematical merges possible
                
                # Apply the best merge dynamically across the local chunk
                new_id = self.merges[best_pair]
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            token_ids.extend(ids)
        return token_ids

    def decode(self, ids):
        """Decode token IDs back to text. Special tokens are silently skipped."""
        special = {self.pad_id, self.bos_id, self.eos_id}
        chunks = []
        for tid in ids:
            if tid in special:
                continue
            chunks.append(self.vocab.get(tid, b'<UNK>'))
        return b''.join(chunks).decode('utf-8', errors='replace')

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        vocab_serial = {str(k): list(v) for k, v in self.vocab.items()}
        merges_serial = {f"{a},{b}": new_id for (a, b), new_id in self.merges.items()}
        data = {
            'vocab': vocab_serial,
            'merges': merges_serial,
            'pad_id': self.pad_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        self.merges = {}
        for pair_str, new_id in data['merges'].items():
            a, b = map(int, pair_str.split(','))
            self.merges[(a, b)] = new_id
        self.merge_list = sorted(self.merges.items(), key=lambda x: x[1])

        self.pad_id = data.get('pad_id')
        self.bos_id = data.get('bos_id')
        self.eos_id = data.get('eos_id')

        # Backward compatibility: map old sep_id to eos_id
        if self.eos_id is None and 'sep_id' in data:
            self.eos_id = data['sep_id']

        return True

    @property
    def vocab_size(self):
        return len(self.vocab)
