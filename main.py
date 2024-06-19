"""CLI interface with the model for testing purposes"""
from interface import ModelInterface

def main():
    model_interface = ModelInterface()
    while True:
        message = input("Enter a message: ")
        toxicity_score = model_interface.analyze_message(message)[1]
        print(f"Toxicity score: {toxicity_score}")

if __name__ == "__main__":
    main()
