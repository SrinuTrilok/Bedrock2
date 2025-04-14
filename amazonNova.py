import boto3
import json

# Set your prompt
prompt_data = "Generate a person working on a laptop in an office"

# ✅ Fix: Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Prepare payload for Amazon Nova
payload = {
    "inferenceConfig": {
        "max_new_tokens": 1000
    },
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": prompt_data
                }
            ]
        }
    ]
}

body = json.dumps(payload)
model_id = "amazon.nova-pro-v1:0"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    raw_body = response.get("body").read()
    print("\n🔹 Raw Response:\n", raw_body)

    parsed = json.loads(raw_body)
    print("\n🔹 Parsed Response:\n", parsed)

    if 'output' in parsed and 'message' in parsed['output']:
        message = parsed['output']['message']
        if 'content' in message and len(message['content']) > 0:
            print("\n🔹 Output Text:\n", message['content'][0]['text'])
        else:
            print("\n No text content found in the message.")
    else:
        print("\n 'output.message' not found in response.")

except Exception as e:
    print("Error invoking model:", e)
