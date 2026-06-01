import streamlit as st  
import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv


load_dotenv()

my_key = os.getenv("GEMINI_API_KEY")

st.title("Mental Health Chatbot")
st.write("Project created using CrewAI and Gemini LLM")

if not my_key:
    st.warning("Please add your GEMINI_API_KEY in the .env file to run the app.")
    st.stop()


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

               
                bot_crew = Crew(
                    agents=[agent1, agent2],
                    tasks=[t1, t2],
                    process=Process.sequential,
                    verbose=False
                )

                bot_crew.kickoff()

               
                st.subheader("Results:")
                
                st.write("**Detected Emotion:**")
                st.info(t1.output.raw)
                
                st.write("**Supportive Response:**")
                st.write(t2.output.raw)

            except Exception as err:
                st.error(f"Error: {err}")