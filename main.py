from crewai.flow.flow import Flow, listen, start
from crewai import LLM, Agent
from pydantic import BaseModel
from crews.mlops_crew.crew import MLOpsCrew
from typing import List
import nbformat

class CodeOutput(BaseModel):
    code: str

class FlowState(BaseModel):
    notebook_path: str = ""
    code: str = ""

class JupyterToTrainFlow(Flow[FlowState]):
    @start()
    def load_code(self):
        nb = nbformat.read(open(self.state.notebook_path), as_version=4)
        code_cells = [cell['source'] for cell in nb.cells if cell.cell_type == 'code']
        self.state.code = "\n\n".join(code_cells)
        return self.state.code
    
    @listen(load_code)
    def refactor_code(self):
        train_code = MLOpsCrew().crew().kickoff(inputs={"code": self.state.code})
        return train_code
    
    @listen(refactor_code)
    def create_req_file(self, train_code):
        class Packages(BaseModel):
            packages: List[str]
            
        llm = LLM(model="ollama/qwen2.5-coder:1.5b",
                  base_url="http://localhost:11434", 
                  timeout=1200)
        agent = Agent(
            role="Python Dependency Extractor",
            goal="Extract Python dependencies from the code. Every package you see in the code, you need to extract it and return it as a list of packages. You will not produce any other text than the list.",
            backstory="You are a Python dependency extractor. You are given a code and you need to extract the dependencies from the code.",
            llm=llm
        )
    
        result = agent.kickoff(
            f"Extract all Python package dependencies (imports) from this code: {train_code}",
            response_format=Packages
            )

        if result.pydantic:
            print("result", result.pydantic)
        else:
            print("result", result)
        # Write to file
        with open("output/requirements.txt", "w") as f:
            f.write("\n".join(result.pydantic.packages))
        return result

def kickoff():
    flow = JupyterToTrainFlow()
    return flow.kickoff(inputs={"notebook_path": 
                                "C:\\Users\\tomme\\OneDrive\\Desktop\\jupyter to training\\jupyter_to_training\\example.ipynb"})


if __name__ == "__main__":
    kickoff()

