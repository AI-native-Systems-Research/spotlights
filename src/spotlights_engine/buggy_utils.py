def add(a: int | float, b: int | float) -> int | float:
    return a + b

def divide(a: int | float, b: int | float) -> float:
    return a / b

def get_average(numbers: list[int | float]) -> float:
    return sum(numbers) / len(numbers)

def find_item(items: list, target) -> int | None:
    for i, item in enumerate(items):
        if item == target:
            return i
    return None

def is_even(number: int) -> bool:
    return number % 2 == 0

def repeat_string(text: str, times: int) -> str:
    return text * times

def concatenate_strings(text1, text2):
    return text1 + text2
