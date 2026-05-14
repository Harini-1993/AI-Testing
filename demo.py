try:
    raw_value = input("Enter a number: ").strip()
    num = int(raw_value)
except ValueError:
    print("Invalid input. Please enter a whole number.")
else:
    if num > 1:
        print("This is valid")
    else:
        print("This is invalid")
