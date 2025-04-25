<h2> Neuro-LIFT: A Neuromorphic, <u>L</u>LM-based <U>I</u>nteractive <u>F</u>ramework for Autonomous Drone Fligh<u>T</u> at the Edge </h2>

This repository contains code associated with Neuro-LIFT: A Neuromorphic, <u>L</u>LM-based <U>I</u>nteractive <u>F</u>ramework for Autonomous Drone Fligh<u>T</u> at the Edge.This code has most recently been tested with Python 3.10 and PyTorch 2.5.1

<h3> Introduction </h3>
The integration of human-intuitive interactions into autonomous systems has been limited. Traditional Natural Language Processing (NLP) systems struggle with context and intent understanding, severely restricting human-robot interaction. Recent advancements in Large Language Models (LLMs) have transformed this dynamic, allowing for intuitive and highlevel communication through speech and text, and bridging the gap between human commands and robotic actions. Additionally, autonomous navigation has emerged as a central focus in robotics research, with artificial intelligence (AI) increasingly being leveraged to enhance these systems. However, existing AI-based navigation algorithms face significant challenges in latency-critical tasks where rapid decision-making is critical. Traditional frame-based vision systems, while effective for highlevel decision-making, suffer from high energy consumption and latency, limiting their applicability in real-time scenarios. Neuromorphic vision systems, combining event-based cameras and spiking neural networks (SNNs), offer a promising alternative by enabling energy-efficient, low-latency navigation. Despite their potential, real-world implementations of these systems, particularly on physical platforms such as drones, remain scarce. In this work, we present Neuro-LIFT, a realtime neuromorphic navigation framework implemented on a Parrot Bebop2 quadrotor. Leveraging an LLM for natural language processing, Neuro-LIFT translates human speech into high-level planning commands which are then autonomously executed using event-based neuromorphic vision and physicsdriven planning. Our framework demonstrates its capabilities in navigating in a dynamic environment, avoiding obstacles, and adapting to human instructions in real-time. Demonstration images of Neuro-LIFT navigating through a moving ring in an indoor setting is provided, showcasing the system’s interactive, collaborative potential in autonomous robotics.

<h3> Installation </h3>
Clone this repository using:

```
git clone git@github.com:amoghj98/neuroLIFT.git
```

Create a conda environment using the provided yaml file as follows:

```
conda env create -n $ENV_NAME --file neurolift.yaml
```
Activate the environment using:

```
conda activate $ENV_NAME
```

<h3> Model Finetuning </h3>
Run file `llm_finetuning.py` using:

```
python llm_finetuning.py
```

<h3> Citations </h3>
If you find this work useful in your research, pleas consider citing: <a href="https://arxiv.org/abs/2407.00931"> Amogh Joshi, Sourav Sanyal and Kaushik Roy, "Neuro-LIFT: A Neuromorphic, LLM-based Interactive Framework for Autonomous Drone FlighT at the Edge", arXiv preprint, 2025 </a>

```
@misc{joshi2025neuroliftneuromorphicllmbasedinteractive,
      title={Neuro-LIFT: A Neuromorphic, LLM-based Interactive Framework for Autonomous Drone FlighT at the Edge}, 
      author={Amogh Joshi and Sourav Sanyal and Kaushik Roy},
      year={2025},
      eprint={2501.19259},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.19259}, 
}
```
and <a href="https://arxiv.org/abs/2407.00931"> Amogh Joshi, Sourav Sanyal and Kaushik Roy, "Real-Time Neuromorphic Navigation: Integrating Event-Based Vision and Physics-Driven Planning on a Parrot Bebop2 Quadrotor", arXiv preprint, 2024 </a>
```
@misc{joshi2024realtimeneuromorphicnavigationintegrating,
      title={Real-Time Neuromorphic Navigation: Integrating Event-Based Vision and Physics-Driven Planning on a Parrot Bebop2 Quadrotor}, 
      author={Amogh Joshi and Sourav Sanyal and Kaushik Roy},
      year={2024},
      eprint={2407.00931},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.00931}, 
}
```

<h3> Authors </h3>
Amogh Joshi, Sourav Sanyal and Kaushik Roy