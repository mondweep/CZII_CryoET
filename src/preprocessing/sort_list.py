def sort_data():
    # Example list with numbers
    numbers = [64, 34, 25, 12, 22, 11, 90]
    
    # Basic sorting (ascending)
    sorted_numbers = sorted(numbers)
    print(f"Original list: {numbers}")
    print(f"Sorted list (using sorted()): {sorted_numbers}")
    
    # In-place sorting using list.sort()
    numbers.sort()
    print(f"In-place sorted list: {numbers}")
    
    # Example with strings
    fruits = ["banana", "apple", "orange", "kiwi", "mango"]
    
    # Sort in ascending order
    fruits.sort()
    print(f"\nSorted fruits (ascending): {fruits}")
    
    # Sort in descending order
    fruits.sort(reverse=True)
    print(f"Sorted fruits (descending): {fruits}")
    
    # Example with custom sorting
    people = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 20},
        {"name": "Charlie", "age": 30}
    ]
    
    # Sort by age
    sorted_by_age = sorted(people, key=lambda x: x["age"])
    print(f"\nPeople sorted by age: {sorted_by_age}")

if __name__ == "__main__":
    sort_data() 