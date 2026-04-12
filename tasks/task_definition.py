TASKS = {
    "easy": {
        "code": """
    def divide(a, b):
        if b == 1:
            return a / b
        return a / b
    """,
        "expected": {
            "issues": [
                "possible division by zero edge case",
                "missing input validation"
            ],
            "severity": "medium",
            "fix_keywords": ["check", "validate", "zero", "guard"],
            "concepts": ["runtime safety", "input validation"]
        }
    },
    "medium": {
        "code": """
    def get_user(id):
        query = "SELECT * FROM users WHERE id = " + str(id)
        log = "fetching user " + id
        return query
    """,
        "expected": {
            "issues": [
                "sql injection risk",
                "string concatenation vulnerability",
                "logging sensitive data improperly"
            ],
            "severity": "high",
            "fix_keywords": ["parameterized", "sanitize", "escape", "log safely"],
            "concepts": ["security", "injection", "data leak"]
        }
    },
    "hard": {
        "code": """
    def process_data(data):
        result = []
        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    result.append(data[i] * data[j])

        if len(result) > 10:
            print("large dataset")

        return sum(result) / (len(result) + 1e-9)
    """,
        "expected": {
            "issues": [
                "inefficient nested loops",
                "unnecessary computation",
                "incorrect normalization in division",
                "debug print in production code",
                "potential performance bottleneck"
            ],
            "severity": "medium",
            "fix_keywords": [
                "optimize",
                "vectorize",
                "remove print",
                "avoid nested loops",
                "use single pass",
                "efficient aggregation"
            ],
            "concepts": [
                "performance",
                "numerical stability",
                "clean code",
                "optimization"
            ]
        }
    }     
}

