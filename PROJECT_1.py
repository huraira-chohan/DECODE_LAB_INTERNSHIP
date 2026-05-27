responses = {
    "hello" : "Hi,how can I help?",
    "hi" : "Hi there!",
    "thanks" : "You're welcome.",
    "how are you" : "Just fine. Are you good?",
    "bye" : "Goodbye!!"
}

print("Hello! Type ""exit"" to quit.")

while True:

    user_inut = input("Message : ")
    clean_input = user_inut.lower().strip()

    if(clean_input == 'exit'):
        print("Bot : Bye!")
        break

    reply = responses.get(clean_input,"I couldn't get it,plz type again.")

    print("Bot : ",reply)

