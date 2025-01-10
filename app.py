from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

app = Flask(__name__)

# Load the Codex model and tokenizer
model_name = "Salesforce/codegen-350M-mono"  # You can choose different sizes
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        code = request.form['code']
        query = request.form['query']
        
        # Start timing
        start_time = time.time()
        
        # Prepare the prompt
        prompt = "Code:\n{}\n\nQuery: {}\n\nModified code:".format(code, query)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response and clean it up
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the modified code part
        try:
            modified_code = response.split("Modified code:")[1].strip()
        except IndexError:
            modified_code = "Error: Could not generate modified code."

        # Calculate execution time
        execution_time = time.time() - start_time
        
        return render_template('result.html', 
                             original_code=code,
                             query=query,
                             modified_code=modified_code,
                             execution_time=f"{execution_time:.2f}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False) 
