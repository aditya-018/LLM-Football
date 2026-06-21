"""Example submission script for the Fortran solution generator.

Requirements:
- Set GROQ_API_KEY in the environment before running.
- Install gfortran (e.g. `brew install gcc`) so Fortran code can compile.
- The API key should be provided externally, not hard-coded.

Usage:
    export GROQ_API_KEY="YOUR_API_KEY_HERE"
    python3 lab86_python.py
"""

import os
import shutil
import subprocess
import tempfile
import time

from groq import Groq

REFERENCE_ANSWER = 37550402023
N = 1_000_000

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY environment variable is not set. Export it before running the script.")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "openai/gpt-oss-20b"

GENERATION_PROMPT = f"""
Generate a COMPLETE Fortran program that calculates
the sum of all prime numbers from 1 to {N}.

Requirements:
- Return ONLY valid Fortran source code.
- Program must print a single integer.
- Use a unique implementation strategy.
- Optimize for either readability, compactness,
  or performance.
"""

REVIEW_PROMPT = """
Review the following Fortran code.

Check:
1. Correctness
2. Performance
3. Compilation issues

Return ONLY an improved Fortran program.
"""

def call_llm(prompt, temperature=0.7):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_completion_tokens=4096,
        top_p=1,
        reasoning_effort="medium",
        stream=False
    )
    return completion.choices[0].message.content.strip()

def generate_solution():
    return call_llm(
        GENERATION_PROMPT,
        temperature=1.2
    )

def review_solution(code):
    prompt = f"""
{REVIEW_PROMPT}
CODE:
{code}
"""
    return call_llm(
        prompt,
        temperature=0.3
    )

def fix_solution(code, error):
    prompt = f"""
The following Fortran program failed.
ERROR:
{error}
CODE:
{code}
Fix the issue.
Return ONLY corrected Fortran code.
"""
    return call_llm(
        prompt,
        temperature=0.2
    )

def compile_fortran(source_file):
    if shutil.which("gfortran") is None:
        return False, (
            "gfortran executable not found. "
            "Install it with Homebrew: `brew install gcc` "
            "or otherwise add gfortran to your PATH."
        )
    executable = source_file.replace(".f90", "")
    result = subprocess.run(
        ["gfortran", "-O3", source_file, "-o", executable],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False, result.stderr
    return True, executable

def execute_program(executable):
    start = time.perf_counter()
    result = subprocess.run(
        [executable],
        capture_output=True,
        text=True,
        timeout=30
    )
    runtime = time.perf_counter() - start
    try:
        output = int(result.stdout.strip())
    except Exception:
        return False, "Invalid output"
    if output != REFERENCE_ANSWER:
        return False, f"Wrong answer: {output}"
    return True, runtime

def validate_solution(code):
    with tempfile.NamedTemporaryFile(
        suffix=".f90",
        delete=False,
        mode="w"
    ) as f:
        f.write(code)
        source_file = f.name
    success, result = compile_fortran(source_file)
    if not success:
        return False, result
    return execute_program(result)

def build_solution():
    code = generate_solution()
    code = review_solution(code)
    for attempt in range(3):
        success, result = validate_solution(code)
        if success:
            runtime = result
            return {
                "code": code,
                "runtime": runtime,
                "score": 1.0 / runtime
            }
        print(
            f"Attempt {attempt + 1} failed:"
            f" {result[:200]}"
        )
        code = fix_solution(
            code,
            result
        )
    return None

def main():
    accepted = []
    while len(accepted) < 10:
        print(
            f"\nGenerating solution "
            f"{len(accepted)+1}/10..."
        )
        result = build_solution()
        if result:
            accepted.append(result)
            print(
                f"Accepted "
                f"runtime={result['runtime']:.6f}s"
            )
    ranked = sorted(
        accepted,
        key=lambda x: x["score"],
        reverse=True
    )
    print("\n===================")
    print("FINAL RANKING")
    print("===================\n")
    for idx, solution in enumerate(
        ranked,
        start=1
    ):
        print(
            f"{idx}. "
            f"Runtime: {solution['runtime']:.6f}s "
            f"Score: {solution['score']:.6f}"
        )
if __name__ == "__main__":
    main()