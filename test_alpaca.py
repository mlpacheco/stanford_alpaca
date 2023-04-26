import argparse
import transformers

def main(args):

    alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(args.path_weights)
    alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(args.path_weights)

    while True:

        input_text = input("Prompt: ")

        inputs = alpaca_tokenizer(input_text, return_tensors='pt')
        out = alpaca_model.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = alpaca_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_weights', type=str, required=True)
    args = parser.parse_args()
    main(args)
