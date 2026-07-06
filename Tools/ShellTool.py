from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools import ShellTool

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_effort="none"
)

tool = ShellTool()

template = PromptTemplate(
    template="""
You are a Linux terminal assistant.

Convert the user's request into EXACTLY ONE executable bash command.

Rules:
- Return ONLY the command.
- No explanations.
- No markdown.
- No extra text.
- Use bash syntax.
- If the user wants to create a file, ALWAYS use a here-document.
- Never use echo or printf to write multi-line files.
- Preserve the file contents exactly.

Example:

mkdir -p <folder> && cat > <folder>/<filename> <<'EOF'
<complete file contents>
EOF

User Request:
{request}
""",
    input_variables=["request"]
)

request = input("What do you want to do?\n")

prompt = template.invoke({
    "request": request
})

response = llm.invoke(prompt)

command = response.content.strip()

# Remove markdown if the model returns it
if command.startswith("```"):
    lines = command.splitlines()

    # Remove opening ``` or ```bash
    lines = lines[1:]

    # Remove closing ```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    command = "\n".join(lines).strip()

print("\nGenerated Command:\n")
print(command)

print("\nExecuting...\n")

try:
    output = tool.invoke(command)
    print(output)
except Exception as e:
    print(f"Execution failed: {e}")