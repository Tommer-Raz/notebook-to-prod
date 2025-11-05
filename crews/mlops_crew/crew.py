from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai import LLM


@CrewBase
class MLOpsCrew():
    """MLOps crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    llm = LLM(model="ollama/qwen3:1.7b",base_url="http://localhost:11434", timeout=1200)

    @agent
    def ml_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['ml_engineer'],
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    @agent
    def qa_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['qa_engineer'],
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    @task
    def refactor_task(self) -> Task:
        return Task(
            config=self.tasks_config['refactor_task'],
            agent=self.ml_engineer_agent(),
            output_file="output/train.py",
            create_directory=True
        )
    
    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config['review_task'],
            agent=self.qa_engineer_agent(),
            output_file="output/refactor.py",
            create_directory=True
        )
        
    @crew
    def crew(self) -> Crew:
        """Creates the MLopsCrew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            llm=self.llm,
            process=Process.sequential,
            verbose=True, 
        )