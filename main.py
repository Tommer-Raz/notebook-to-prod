from json import load
from crewai.flow.flow import Flow, listen, start
from crewai.agent import Agent
from crewai import LLM
from pydantic import BaseModel, Field
from crews.mlops_crew.crew import MLOpsCrew
import nbformat

# Define flow state
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
        result = MLOpsCrew().crew().kickoff(inputs={"code": self.state.code})
        print(result)
        return result

def kickoff():
    flow = JupyterToTrainFlow()
    return flow.kickoff(inputs={"notebook_path": 
                                "C:\\Users\\tomme\\OneDrive\\Desktop\\jupyter to training\\jupyter_to_training\\example.ipynb"})


if __name__ == "__main__":
    kickoff()
