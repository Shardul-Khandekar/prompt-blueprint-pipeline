import os
import json
from openai import OpenAI

# Retrieve OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Raise an error if the API key is not set
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set in GitHub Actions")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load prompt and test data
with open("prompts/summarize_article.txt", "r") as file:
    prompt_template = file.read()

with open("tests/test_data.json", "r") as file:
    test_cases = json.load(file)

print(f"Running smoke tests with {len(test_cases)} test cases")
test_failed = False

# Iterate through each test case
for case in test_cases:
    print(f"Running test case: {case['id']}")

    # Hydrate prompt with test case data
    hydrated_prompt = prompt_template.replace("{article}", case["input"])

    print(f"Hydrated Prompt: {hydrated_prompt}")

    # Call OpenAI API to get the summary
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": hydrated_prompt}
            ]
        )
        output = response.choices[0].message.content.strip()
        print(f"API Output: {output}")

        if case["expected_keyword"].lower() in output.lower():
            print(
                f"PASS: Expected keyword '{case['expected_keyword']}' found in output")
        else:
            print(
                f"FAIL: Expected keyword '{case['expected_keyword']}' NOT found in output")
            test_failed = True
    
    except Exception as e:
        print(f"FAIL: API call failed with error: {e}")
        test_failed = True

if test_failed:
    raise AssertionError("Some smoke tests failed. Check logs for details.")
    exit(1)
else:
    print("All smoke tests passed successfully.")
    exit(0)