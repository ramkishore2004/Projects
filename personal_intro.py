# --- 1. Get User Information ---

# Get the user's name and store it in the 'name' variable
name = input("Hello! Welcome to the Introduction Program. What is your name? ")

# Get the user's age and store it in the 'age' variable
# Note: input() gets the age as a string, which is fine for this program
age = input("It's nice to meet you, " + name + "! How old are you? ")

# Get the user's hobby and store it in the 'hobby' variable
hobby = input("That's interesting! Finally, what is a favorite hobby of yours? ")

# --- 2. Display Friendly Welcome Message ---

print("\n--- Program Output ---")
print("🎉 Welcome, " + name + "! 🎉")
print("We've gathered all your information, and here is your friendly introduction:")
print("------------------------------------------")

# Display the stored information in a welcoming format
print("It's wonderful to have you here! You are **" + name + "**.")
print("At the age of **" + age + "**, you enjoy **" + hobby + "**.")
print("What a fantastic combination! We hope you have a great time.")
print("------------------------------------------")