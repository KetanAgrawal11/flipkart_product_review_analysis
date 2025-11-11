import os
import subprocess

def run_script(script_path):
    print(f"\nRunning {script_path} ...")
    os.system(f"python {script_path}")

def main():
    while True:
        print("\nProduct Review Analysis Pipeline")
        print("===================================")
        print("Choose a phase to run:")
        print("1️) Preprocessing & Cleaning")
        print("2️) POS + NER Analysis")
        print("3️) Sentiment Analysis")
        print("4️) Word Similarity & Semantics")
        print("5️) Interactive QA System")
        print("6️) Summarization")
        print("7️) Run Full Pipeline (All Steps)")
        print("0️) Exit")


        choice = input("\nEnter your choice: ").strip()

        scripts = {
            "1": "flipkart_product-review-analysis/src/preprocessing/clean_translate.py",
            "2": "flipkart_product-review-analysis/src/analysis/pos_ner_analysis.py",
            "3": "flipkart_product-review-analysis/src/analysis/sentiment_analysis.py",
            "4": "flipkart_product-review-analysis/src/analysis/vector_semantics.py",
            "5": "flipkart_product-review-analysis/src/analysis/question_answering.py",
            "6": "flipkart_product-review-analysis/src/summarization/review_summarization.py"
        }

        if choice == "7":
            for script in list(scripts.values())[:-1]:  # run all except summarization
                run_script(script)
            run_script(scripts["6"])  # summarization last
        elif choice in scripts:
            run_script(scripts[choice])
        else:
            print("Exiting...")
            break 


if __name__ == "__main__":
    main()
