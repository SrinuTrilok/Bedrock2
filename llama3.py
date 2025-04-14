import boto3
import json

prompt_data = "Write a poem on Anish"
bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt": prompt_data,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}
body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

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

    # Try extracting "generation" if it exists
    if 'generation' in parsed:
        print("\n🔹 Output Text:\n", parsed['generation'])
    else:
        print("\n⚠️ 'generation' key not found in response")

except Exception as e:
    print("Error invoking model:", e)
