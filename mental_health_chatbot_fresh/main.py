import streamlit as st  
import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# Yeh line .env file se key load karegi
load_dotenv()

# Ab aapki key system environment se automatic uth jayegi
my_key = os.getenv("GEMINI_API_KEY")

# UI design - clean and student project style
st.title("Mental Health Chatbot")
st.write("Project created using CrewAI and Gemini LLM")

if not my_key:
    st.warning("Please add your GEMINI_API_KEY in the .env file to run the app.")
    st.stop()

# Fixed Gemini config using standard LiteLLM string updated to stable 2.5 version
my_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=my_key
)

user_msg = st.text_input("Enter your message here:")

if st.button("Submit"):
    if user_msg == "":
        st.write("Please write something first.")
    else:
        with st.spinner("Processing..."):
            try:
                # Making 2 agents for project (Kept simple and practical for submission)
                agent1 = Agent(
                    role="Emotion Detector",
                    goal="Find the main emotion from user text. Give answer in one word with emoji.",
                    backstory="You are good at finding mood from text.",
                    llm=my_llm,
                    verbose=False
                )

                agent2 = Agent(
                    role="Support Agent",
                    goal="Give a helpful and kind response to the user according to their problem.",
                    backstory="You are a nice counselor who helps people.",
                    llm=my_llm,
                    verbose=False
                )

                # Creating tasks
                t1 = Task(
                    description=f"Find emotion: {user_msg}",
                    expected_output="One word emotion name like Sad, Happy, Angry etc with emoji.",
                    agent=agent1
                )

                t2 = Task(
                    description=f"Give supportive lines for: {user_msg}",
                    expected_output="A kind supportive paragraph.",
                    agent=agent2
                )

                # Running crewai logic
                bot_crew = Crew(
                    agents=[agent1, agent2],
                    tasks=[t1, t2],
                    process=Process.sequential,
                    verbose=False
                )

                bot_crew.kickoff()

                # Showing final results
                st.subheader("Results:")
                
                st.write("**Detected Emotion:**")
                st.info(t1.output.raw)
                
                st.write("**Supportive Response:**")
                st.write(t2.output.raw)

            except Exception as err:
                st.error(f"Error: {err}")