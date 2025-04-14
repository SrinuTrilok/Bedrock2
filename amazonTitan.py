import boto3
import base64
import json
import re
from pathlib import Path
 
# Titan client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
 
# Prompt text
prompt_text = "Amzon Bedrock Services"
# Full request body
prompt = {
    "textToImageParams": {
        "text": prompt_text
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 8,
        "seed": 42,
        "quality": "standard",
        "width": 1024,
        "height": 1024,
        "numberOfImages": 1
    }
}
 
# Make the API call
response = bedrock.invoke_model(
    modelId="amazon.titan-image-generator-v1",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(prompt)
)
 
# Decode image
response_body = json.loads(response["body"].read())
image_data = base64.b64decode(response_body["images"][0])
 
# Clean filename from prompt text
def clean_filename(text):
    # Replace spaces with underscores, remove non-alphanumeric
    return re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_"))[:50]
 
filename = clean_filename(prompt_text) + ".png"
 
# Ensure output folder exists
output_dir = Path("generated_images")
output_dir.mkdir(parents=True, exist_ok=True)
 
# Write to file
output_path = output_dir / filename
with open(output_path, "wb") as f:
    f.write(image_data)
 
print(f"✅ Image saved at: {output_path}")