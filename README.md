<h1 align="center">Sentiment Analysis in Change Management Using Zero-Shot Models</h1>

---

## Project Objective
This project aims to automate the classification of employee reactions to organizational changes by leveraging natural language processing algorithms. Our sentiment analysis application seeks to identify the level of employee engagement with proposed changes.

---

## Context 
In the context of organizational change management, analyzing written and verbal communications from employees is essential. However, the lack of labeled data has led us to use zero-shot classification models. These models, which do not require labeled data for specific classes, provide great flexibility and adaptability for unseen tasks.

---

## Proposed Solution
We use Large Language Models (LLMs), available as open-source on the Hugging Face platform, and adapt them to our task using prompting techniques. A comparative evaluation of these models was conducted to identify the best-performing one for our specific application.

---

## Files and Directories
- **Models_Zero_Shot_Evaluation.csv**: Contains evaluation results of the models across various benchmarks.
- **Gestion des changements.csv**: Evaluation dataset containing diverse excerpts from employee communications.
- **Models_SA.ipynb**: Jupyter Notebook for the comparative evaluation of the models.
- **streamlit_App/App.py**: Streamlit application developed for interactive results visualization.
- **streamlit_App/plot_utils.py**: Contains plotting tools and other functions used in the Streamlit application.
- **requirements.txt**: List of libraries required to run the application.

---

## Application Deployment
To run the Streamlit application locally, follow these steps:

1. Clone the repository to your local machine.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
3. Navigate to the streamlit_App folder and launch the application :
    ```bash
   streamlit run App.py
**Link to the Streamlit application** : https://bxzfe8urqtecurcmrs9red.streamlit.app/
