import os
import json
from openai import OpenAI
import re

generator_model = "gpt-3.5-turbo"
evaluator_model = "gpt-4o"
# Minimum score 3 required to pass the evaluation
score_threshold = 3

# Retrieve OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Raise an error if the API key is not set
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set in GitHub Actions")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load all files needed for the evaluation tests
try:
    with open("prompts/summarize_article.txt", "r") as f:
        prompt_template = f.read()

    with open("tests/evaluation_prompt.txt", "r") as f:
        evaluation_template = f.read()

    with open("tests/test_data.json", "r") as f:
        test_cases = json.load(f)

except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}")
    exit(1)

print(f"Loaded {len(test_cases)} test case(s).")
test_results = []


def run_quality_test(case, hydrated_prompt):
    """
    Runs the LLM as evaluator for quality test
    """
    try:
        response = client.chat.completions.create(
            model=generator_model,
            messages=[{"role": "user", "content": hydrated_prompt}]
        )
        actual_output = response.choices[0].message.content
        print(f"Model Output:\n{actual_output}")

        eval_prompt_hydrated = evaluation_template.replace(
            "{input}", case.get("input", ""))
        eval_prompt_hydrated = eval_prompt_hydrated.replace(
            "{ideal_output}", case.get("ideal_output", ""))
        eval_prompt_hydrated = eval_prompt_hydrated.replace(
            "{output}", actual_output)

        # Capture JSON output from evaluator
        eval_response = client.chat.completions.create(
            model=evaluator_model,
            messages=[{"role": "user", "content": eval_prompt_hydrated}],
            response_format={"type": "json_object"}
        )

        eval_json_str = eval_response.choices[0].message.content

        # Parse the JSON evaluation
        evaluation = json.loads(eval_json_str)
        print(f"Evaluation Scores: {evaluation}")

        # Check scores against our threshold
        acc_score = evaluation.get("accuracy_score", 0)
        con_score = evaluation.get("conciseness_score", 0)

        if acc_score < score_threshold or con_score < score_threshold:
            print(
                f"FAIL: Accuracy score ({acc_score}) is below threshold ({score_threshold}).")
            print(f"Reason: {evaluation.get('accuracy_reasoning')}")
            return False
        else:
            print(f"PASS: Accuracy score: {acc_score}")
            return True
    except Exception as e:
        print(f"FAIL: Test execution error: {e}")
        return False


def run_format_test(case, hydrated_prompt):
    """
    Runs a simple regex-based format test.
    """
    print("Running 'format' test...")
    try:
        response = client.chat.completions.create(
            model=generator_model,
            messages=[{"role": "user", "content": hydrated_prompt}]
        )
        actual_output = response.choices[0].message.content.strip()
        print(f"Model Output:\n{actual_output}")

        if case["expected_format"] == "starts_with_bullet":
            # Regex to check for common bullet point markers (•, *, -)
            # at the very beginning of the string (^)
            if re.match(r"^[•*-]", actual_output):
                print("PASS: Output correctly starts with a bullet point.")
                return True
            else:
                print("FAIL: Output does not start with a bullet point.")
                return False

        print(f"FAIL: Unknown format type '{case['expected_format']}'")
        return False

    except Exception as e:
        print(f"FAIL: Test execution error: {e}")
        return False


# Iterate through each test case
for case in test_cases:
    print(f"Running test case: {case['id']}")

    # Update the prompt template with the test case input
    try:
        hydrated_prompt = prompt_template.format(article_text=case["input"])

    except Exception as e:
        print(
            f"FAIL: API call failed for evaluator model {evaluator_model}. Error: {e}")
        tests_failed = True

if tests_failed:
    print("\n--- Evaluation FAILED. ---")
    exit(1)
else:
    print("\n--- All evaluations PASSED. ---")
    exit(0)
