import pickle


class TokenizerChar:
    def __init__(self, meta_path: str) -> None:
        
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)

    def stoi(self, token: str) -> int:
        r"""E.g., 'a' -> 39.
        """
        return self.meta["stoi"][token]

    def itos(self, id: int) -> str:
        r"""E.g., 39 -> 'a'.
        """
        return self.meta["itos"][id]

    def __len__(self) -> int:
        return self.meta["vocab_size"]