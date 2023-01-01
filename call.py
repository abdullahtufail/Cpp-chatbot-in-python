import nltk
nltk.download('punkt')

from chatbotpoly import MyChatbot

state = True
print("C++ CHATBOT, Press >quit< to exit...")
chatbot = MyChatbot('advanced')
greeting = chatbot.greet()
print(greeting)  # Output: "Hello, I am My Chatbot. How can I help you today?"
while state:
    print("YOU: ")
    msg = input()
    if msg.lower() == "quit":
        state = False
        break
    res = chatbot.chatbot_response(msg)
    print("BOT: ", res)
