import random


class MySampler:
    def __init__(self, books_num: int):
        self.books_num = books_num
        
    def __iter__(self) -> int:
        while True:
            yield random.randint(a=0, b=self.books_num)