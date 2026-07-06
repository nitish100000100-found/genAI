from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class AddInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers."

    args_schema: Type[BaseModel] = AddInput

    def _run(self, a: float, b: float):
        return a + b

    async def _arun(self, a: float, b: float):
        return self._run(a, b)

tool = AddTool()

print(tool.invoke({
    "a": 10,
    "b": 20
}))

##########################

from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

print(add.invoke({
    "a": 10,
    "b": 20
}))



from pydantic import BaseModel, Field
from langchain_core.tools import tool

class AddInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

@tool(args_schema=AddInput)
def add(a: float, b: float):
    """Adds two numbers."""
    return a + b




#########################


from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class AddInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

def add(a: float, b: float):
    return a + b

tool = StructuredTool.from_function(
    func=add,
    args_schema=AddInput,
    description="Adds two numbers."
)



########################################

from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtracts two numbers."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


# Toolkit
toolkit = [
    add,
    subtract,
    multiply,
]

# Example
print(toolkit[0].invoke({"a": 10, "b": 5}))  # 15
print(toolkit[1].invoke({"a": 10, "b": 5}))  # 5
print(toolkit[2].invoke({"a": 10, "b": 5}))  # 50