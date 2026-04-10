TASKS = {

    "easy": {
        "code": """
            def divide(a, b):
                return a / b
            """,
        "expected": {
            "issues": ["division by zero"],
            "severity": "medium",
            "fix_keywords": ["check", "zero", "if"],
            "concepts": ["runtime error"]
        }
    },

    "medium": {
        "code": """
            def get_user(id):
                query = "SELECT * FROM users WHERE id = " + id
                return query
            """,
        "expected": {
            "issues": ["sql injection"],
            "severity": "high",
            "fix_keywords": ["parameterized", "prepared"],
            "concepts": ["security", "injection"]
        }
    },

    "hard": {
        "code": """
            def process_data(data):
                result = []
                for i in range(len(data)):
                    for j in range(len(data)):
                        result.append(data[i] * data[j])

                print("Processing done:", result)

                if len(result) > 0:
                    return sum(result) / len(result)
                else:
                    return 0
            """,
        "expected": {
            "issues": [
                "quadratic time complexity",
                "high memory usage",
                "unnecessary computation",
                "printing side effect"
            ],
            "severity": "medium",
            "fix_keywords": [
                "remove",
                "print",
                "optimize",
                "avoid",
                "loop",
                "compute directly",
                "sum",
                "mean"
            ],
            "concepts": [
                "performance",
                "time complexity",
                "clean code"
            ]
        }
    }
    
}

