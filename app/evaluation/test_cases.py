"""
Evaluation dataset — questions, ground truth answers, and the role needed to access the answer.
Ground truths are derived directly from the source documents in data/documents/.
"""

# Each case: question, ground_truth, user_role
# ground_truth is the ideal answer a perfect RAG system would return.
# user_role controls which ChromaDB chunks are visible during retrieval.

EVAL_CASES = [
    # --- General (accessible by all employees) ---
    {
        "question": "What are the standard working hours at Nexora?",
        "ground_truth": "Standard working hours are 9:30 AM to 6:30 PM, Monday through Friday. Core hours during which all employees must be available are 11:00 AM to 4:00 PM.",
        "user_role": "employee",
    },
    {
        "question": "How many annual leave days do employees with 0 to 2 years of service get?",
        "ground_truth": "Employees with 0 to 2 years of service receive 18 days of annual leave per year.",
        "user_role": "employee",
    },
    {
        "question": "What is the attendance marking deadline?",
        "ground_truth": "Employees are expected to mark attendance by 10:00 AM via the HR portal at hrportal.nexora.internal.",
        "user_role": "employee",
    },
    {
        "question": "What happens if an employee has three or more unexcused late arrivals in a month?",
        "ground_truth": "Three or more unexcused late arrivals in a calendar month will trigger a formal discussion with HR.",
        "user_role": "employee",
    },

    # --- HR role ---
    {
        "question": "When is payroll processed each month?",
        "ground_truth": "Payroll is processed on the 25th of each month. If the 25th falls on a weekend or public holiday, payroll is processed on the preceding working day.",
        "user_role": "hr",
    },
    {
        "question": "What is the payroll cut-off date for attendance and leave data?",
        "ground_truth": "The payroll cut-off for attendance and leave data is the 22nd of each month. Any corrections after this date are adjusted in the following month's payroll.",
        "user_role": "hr",
    },

    # --- Finance / C-Level ---
    {
        "question": "What was Nexora's total revenue in Q1 FY2025?",
        "ground_truth": "Nexora Technologies delivered total revenue of ₹18.4 crore in Q1 FY2025, representing 23% year-on-year growth compared to Q1 FY2024.",
        "user_role": "finance",
    },
    {
        "question": "What was the net profit and EBITDA margin in Q1 FY2025?",
        "ground_truth": "Net profit stood at ₹3.2 crore with an EBITDA margin of 24.6% in Q1 FY2025.",
        "user_role": "finance",
    },

    # --- Negative test: question outside the knowledge base ---
    {
        "question": "What is the company's policy on pet insurance?",
        "ground_truth": "I don't have that information in the company knowledge base.",
        "user_role": "employee",
    },
]
