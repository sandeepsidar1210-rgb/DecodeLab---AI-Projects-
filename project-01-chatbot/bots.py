# So I am starting my decodelab internship -- The Project 1 
# Rule Based AI Chatbot


def run_bot():

    responses = {

        # Greetings
        "hello" : "Hey ! I am LogicBot. How can i Help you Today ?",
        "hi" : " Hi there! Type 'help' to see what I can do.",
        "hey" : " Hey! What's up? Type 'help' for commands ",

        # About 
        "who are you": "I am LogicBot — a rule-based AI chatbot built on pure logic.",
        "what are you": "I am a deteministic chatbot. ",
        "waht is your name": "My name is LogicBot. Nice to meet you.",


        # AI related questions
        "what is ai": "AI is the simulation of human intelligence by machines using logic, data, and learning.",
        "what is ml": "ML is a subset of AI where systems learn from data instead of following hardcoded rules.",
        "rule based ai": "Rule-based AI responds using predefined if-else logic or dictionary lookups — like me!",
        "what is a chatbot": "A chatbot is a program that simulates conversation. I'm one of them!",
    

        #  Utility
        "joke": "why milk should not be used during doing something good ? coz' * subh kam mein dairy kaisi !* ",
        "help": "Commands: hello | who are you | what is ai | what is ml | joke | bye | exit",
        "smart": "Yes I am very smart now stop asking me that question !",

        # Farewell
        "bye": "Goodbye! Keep building cool things.",
        "thanks": "It was nice helping you.",
        "thank you": " * Dosti mein no sorry no thankyou * ",
        "goodbye": "I am not leaving this world rn :) " ,
    }

    # Defining Fallbacks

    fallback = "I don't understand that yet. Type 'help' to see valid commands."

    print("-" * 50)
    print("       LogicBot — Rule-Based AI Chatbot")
    print("     Type 'exit' or 'quit' to shut down.")
    print("-" * 50)

    # infinte loop
    while True:
        raw = input ("\nUser:")
        
        # normalization/sanitization - all letters to lowercase basically
        clean = raw.lower().strip()

        # exit 
        if clean in ("exit" , "quit" ):
            print("Bot : Shutting down. GoodBye, User.")
            break

        # imp - dictionary lookup with fallback 
        reply = responses.get(clean , fallback)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    run_bot()
       