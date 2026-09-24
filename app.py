import os, re, subprocess, tempfile
from typing import TypedDict, List, Optional
from flask import Flask, render_template, request
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    testbench: Optional[str]
    report: Optional[str]

def text_of(response):
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            str(x.get("text", "")) if isinstance(x, dict) else str(x)
            for x in content
        )
    return str(content)

def clean_code(text):
    text = text.strip()
    m = re.search(r"```(?:verilog|systemverilog|sv|v)?\s*(.*?)```",
                  text, re.I | re.S)
    return m.group(1).strip() if m else text.replace("```verilog","").replace("```","").strip()

@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 Verilog test scenarios for the given task."""
    response = llm.invoke(f"""
You are a Senior RTL Verification Engineer.
Generate 3 to 5 specific test scenarios for this Verilog task:
{task_description}
Include normal, boundary, and edge cases. Return only a numbered list.
""")
    return text_of(response)

@tool
def run_verilog_simulation(verilog_code: str, testbench_code: str) -> str:
    """Compile and simulate Verilog code using Icarus Verilog."""
    with tempfile.TemporaryDirectory() as d:
        design = os.path.join(d, "design.v")
        tb = os.path.join(d, "testbench.v")
        out = os.path.join(d, "simulation.out")
        open(design, "w", encoding="utf-8").write(verilog_code)
        open(tb, "w", encoding="utf-8").write(testbench_code)

        c = subprocess.run(
            ["iverilog", "-g2012", "-o", out, design, tb],
            capture_output=True, text=True, timeout=30
        )
        if c.returncode:
            return "COMPILATION FAILED\n\n" + (c.stderr or c.stdout)

        r = subprocess.run(
            ["vvp", out], capture_output=True, text=True, timeout=30
        )
        if r.returncode:
            return "SIMULATION FAILED\n\n" + (r.stderr or r.stdout)

        return "COMPILATION: PASS\nSIMULATION: PASS\n\n" + (
            r.stdout or "Simulation completed."
        )

def developer(state):
    task = state["messages"][-1].content
    response = llm.invoke(f"""
You are a professional Verilog RTL designer and verification engineer.

Task:
{task}

Generate a synthesizable Verilog/SystemVerilog design AND a self-checking
Verilog testbench. Include clock/reset when required, meaningful tests,
$display/$monitor, and $finish.

Return exactly:
DESIGN:
<design code>
TESTBENCH:
<testbench code>
Do not add explanations.
""")
    s = text_of(response)
    dm = re.search(r"DESIGN:\s*(.*?)\s*TESTBENCH:", s, re.I | re.S)
    tm = re.search(r"TESTBENCH:\s*(.*)", s, re.I | re.S)
    if not dm or not tm:
        raise ValueError("Gemini did not return DESIGN/TESTBENCH sections.")
    return {"code": clean_code(dm.group(1)),
            "testbench": clean_code(tm.group(1)),
            "next_step": "tester"}

def tester(state):
    cases = generate_test_cases.invoke(state["messages"][-1].content)
    sim = run_verilog_simulation.invoke({
        "verilog_code": state["code"],
        "testbench_code": state["testbench"]
    })
    return {
        "report": "### SIMULATION REPORT\n\n" + sim +
                  "\n\n### TEST SCENARIOS\n\n" + cases,
        "next_step": "manager"
    }

def manager(state):
    return {"next_step": "archiver"}

def archiver(state):
    return {"next_step": "finished"}

g = StateGraph(CrewState)
g.add_node("developer", developer)
g.add_node("tester", tester)
g.add_node("manager", manager)
g.add_node("archiver", archiver)
g.add_edge(START, "developer")
g.add_edge("developer", "tester")
g.add_edge("tester", "manager")
g.add_edge("manager", "archiver")
g.add_edge("archiver", END)
workflow = g.compile()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    task = ""
    if request.method == "POST":
        task = request.form.get("task", "").strip()
        if not task:
            error = "Please enter a Verilog task."
        else:
            try:
                state = workflow.invoke({
                    "messages": [HumanMessage(content=task)]
                }, config={"recursion_limit": 20})
                result = {
                    "code": state.get("code", ""),
                    "testbench": state.get("testbench", ""),
                    "report": state.get("report", "")
                }
            except Exception as e:
                error = str(e)
    return render_template("index.html", result=result, error=error, task=task)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
