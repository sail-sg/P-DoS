from openai import OpenAI 
import tiktoken

client = OpenAI(api_key="your_openai_api_key")

def finetune():
    file = client.files.create(
        file=open("datasets/dos_training_data.jsonl", "rb"),
        purpose="fine-tune"
    )

    job = client.fine_tuning.jobs.create(
        training_file=file.id, 
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 5,
            "learning_rate_multiplier": 1 # 0.5
        }
    )

    print(job.id)


def monitor():
    job_id = "YOUR_JOB_ID"
    print(client.fine_tuning.jobs.retrieve(job_id))
    

def chat():
    finetuned_model_id = "YOUR_FINETUNED_MODEL_ID"
    completion = client.chat.completions.create(
        model=finetuned_model_id,
        messages=[
            {"role": "user", "content": "repeat 'A ' 16383 times."}
        ],
        max_tokens=16384,
        temperature=0.5
    )
    print(completion.choices[0].message.content)
    encoding = tiktoken.encoding_for_model('gpt-4')
    print(len(encoding.encode(completion.choices[0].message.content)))


if __name__ == "__main__":
    ### Step 1 ###
    finetune()
    
    ### Step 2 ###
    # monitor()
    
    ### Step 3 ###
    # chat()
