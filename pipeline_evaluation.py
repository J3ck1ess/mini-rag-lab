import json
from datetime import datetime, timezone
from pathlib import Path
from rag_pipeline import answer_question

# 1. Define the evaluation dataset
evaluation_dataset = [
    # Category 1: Answerable — single facts
    {
        "id": "Q1",
        "category": "answerable",
        "query": "How many days of paid annual leave do employees receive?",
        "expected_points": ["15 days"],
    },
    {
        "id": "Q2",
        "category": "answerable",
        "query": "Who must approve an annual leave application?",
        "expected_points": ["direct supervisor"],
    },
    {
        "id": "Q3",
        "category": "answerable",
        "query": "Who is eligible for the company's health insurance?",
        "expected_points": ["full-time employees"],
    },
    {
        "id": "Q4",
        "category": "answerable",
        "query": "When does health insurance coverage begin?",
        "expected_points": ["probation period"],
    },
    {
        "id": "Q5",
        "category": "answerable",
        "query": "Which platform must employees use to submit remote work requests?",
        "expected_points": ["company portal"],
    },

    # Category 2: Multi-condition — multiple facts
    {
        "id": "Q6",
        "category": "multi_condition",
        "query": "What are the requirements for applying for annual leave?",
        "expected_points": [
            "six months",
            "3 working days",
            "direct supervisor",
        ],
    },
    {
        "id": "Q7",
        "category": "multi_condition",
        "query": "What are the requirements for working remotely?",
        "expected_points": [
            "direct supervisor",
            "company portal",
        ],
    },
    {
        "id": "Q8",
        "category": "multi_condition",
        "query": "What does the company provide regarding health insurance, and when does coverage begin?",
        "expected_points": [
            "full-time employees",
            "probation period",
        ],
    },
    {
        "id": "Q9",
        "category": "multi_condition",
        "query": "What information does the document provide about annual leave?",
        "expected_points": [
            "15 days",
            "six months",
            "3 working days",
            "direct supervisor",
        ],
    },

    # Category 3: Unanswerable — unsupported information
    {
        "id": "Q10",
        "category": "unanswerable",
        "query": "What is the company's parental leave policy?",
        "expected_points": [],
    },
    {
        "id": "Q11",
        "category": "unanswerable",
        "query": "What is the company's annual salary bonus?",
        "expected_points": [],
    },
    {
        "id": "Q12",
        "category": "unanswerable",
        "query": "How much does the company pay for each employee's health insurance?",
        "expected_points": [],
    },
    {
        "id": "Q13",
        "category": "unanswerable",
        "query": "How many days of paid sick leave do employees receive?",
        "expected_points": [],
    },

    # Category 4: Robustness — paraphrased or potentially ambiguous questions
    {
        "id": "Q14",
        "category": "robustness",
        "query": "After joining the company, when can a full-time employee start receiving health insurance coverage?",
        "expected_points": ["probation period"],
    },
    {
        "id": "Q15",
        "category": "robustness",
        "query": "Can employees submit a remote work request without getting approval from their direct supervisor?",
        "expected_points": ["direct supervisor"],
    },
]


# 2. Define answer-checking rules
def evaluate_answer(answer, test_case):
    answer_lower = answer.lower()
    category = test_case["category"]
    expected_points = test_case["expected_points"]

    if category in ("answerable", "multi_condition", "robustness"):
        missing_points = [
            point
            for point in expected_points
            if point.lower() not in answer_lower
        ]

        return {
            "passed": len(missing_points) == 0,
            "missing_points": missing_points,
            "behavior": "answer",
        }

    elif category == "unanswerable":
        refusal_phrase = (
            "i cannot answer this question based on the provided context."
        )
        passed = refusal_phrase in answer_lower

        return {
            "passed": passed,
            "missing_points": [],
            "behavior": "refuse",
        }

    else:
        raise ValueError(f"Unknown category: {category}")


# 3. Run the end-to-end evaluation
def run_evaluation():
    test_cases = evaluation_dataset

    total_cases = len(test_cases)
    passed_cases = 0
    results = []

    for test_case in test_cases:
        query = test_case["query"]

        print("\n" + "=" * 80)
        print(f"Test ID: {test_case['id']}")
        print(f"Category: {test_case['category']}")
        print(f"Question: {query}")

        # Run the complete RAG pipeline.
        answer = answer_question(query)

        # Evaluate the generated answer.
        evaluation_result = evaluate_answer(answer, test_case)

        if evaluation_result["passed"]:
            passed_cases += 1

        print(f"\nActual answer:\n{answer}")
        print(
            f"\nExpected behavior: "
            f"{evaluation_result['behavior']}"
        )
        print(
            f"Result: "
            f"{'PASS' if evaluation_result['passed'] else 'FAIL'}"
        )

        if evaluation_result["missing_points"]:
            print(
                "Missing expected points: "
                + ", ".join(evaluation_result["missing_points"])
            )

        # Store the result for the JSON report.
        results.append({
            "id": test_case["id"],
            "category": test_case["category"],
            "query": query,
            "expected_points": test_case["expected_points"],
            "expected_behavior": evaluation_result["behavior"],
            "actual_answer": answer,
            "passed": evaluation_result["passed"],
            "missing_points": evaluation_result["missing_points"],
        })

    # Calculate the overall metrics.
    failed_cases = total_cases - passed_cases
    pass_rate = (
        passed_cases / total_cases * 100
        if total_cases > 0
        else 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "passed": passed_cases,
        "failed": failed_cases,
        "pass_rate": round(pass_rate, 2),
        "results": results,
    }

    # Create the output directory if necessary.
    report_dir = Path("evaluation_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "pipeline_evaluation.json"

    # Write the report as UTF-8 JSON.
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Print the summary.
    print("\n" + "=" * 80)
    print("=== Pipeline Evaluation Summary ===")
    print(f"Total test cases: {total_cases}")
    print(f"Passed: {passed_cases}")
    print(f"Failed: {failed_cases}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print(f"JSON report: {report_path}")


if __name__ == "__main__":
    run_evaluation()
