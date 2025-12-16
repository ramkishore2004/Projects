# Function to determine grade and message
def get_grade_message(marks):
    if 90 <= marks <= 100:
        return "A", "Excellent work!"
    elif 80 <= marks < 90:
        return "B", "Great job!"
    elif 70 <= marks < 80:
        return "C", "Good effort!"
    elif 60 <= marks < 70:
        return "D", "Keep improving!"
    else:
        return "F", "Needs more practice."

# Main program
while True:
    name = input("Enter student name: ")
    try:
        marks = int(input("Enter marks (0-100): "))
        if 0 <= marks <= 100:
            grade, message = get_grade_message(marks)
            print(f"Student: {name}\nGrade: {grade}\nMessage: {message}")
            break
        else:
            print("Marks must be between 0 and 100. Try again.")
    except ValueError:
        print("Invalid input. Please enter numeric marks.")