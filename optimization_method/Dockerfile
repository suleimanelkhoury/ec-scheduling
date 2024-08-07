FROM python:3

# Install required packages
RUN pip3 install deap Flask matplotlib redis kubernetes numpy scipy pandas optuna docker

RUN mkdir -p /logs/plots

ADD NSGA2.py /NSGA2.py
ADD PSO.py /PSO.py
ADD CMAES.py /CMAES.py
ADD Strategy.py /Strategy.py
ADD main.py /main.py
ADD configuration.py /configuration.py
ADD genotyp_phenotyp.py /genotyp_phenotyp.py


ENTRYPOINT ["python3","-u", "main.py"]
