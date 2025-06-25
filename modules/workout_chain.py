import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

class FitnessPlanChain(Chain):
    retriever: Any
    llm_chain: Any

    class Config:
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["age", "gender", "fitness_goal", "experience_level", "available_equipment", "health_conditions"]

    @property
    def output_keys(self) -> List[str]:
        return ["plan", "source_documents"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        age = inputs.get("age")
        gender = inputs.get("gender", "").strip()
        fitness_goal = inputs.get("fitness_goal", "").strip()
        experience_level = inputs.get("experience_level", "").strip()
        available_equipment = inputs.get("available_equipment", "").strip()
        health_conditions = inputs.get("health_conditions", "").strip()

        if not all([age, gender, fitness_goal, experience_level]):
            raise ValueError("Required inputs cannot be empty")
        if not isinstance(age, int) or age <= 0:
            raise ValueError("Age must be a positive integer")

        query = (
            f"Generate plan for age={age}, gender={gender}, "
            f"goal={fitness_goal}, experience={experience_level}, "
            f"equipment={available_equipment}, conditions={health_conditions}"
        )

        docs = self.retriever.invoke(query)
        unique_docs = []
        seen_names = set()
        for doc in docs:
            name = doc.metadata.get("name")
            if name not in seen_names:
                seen_names.add(name)
                unique_docs.append(doc)
            if len(unique_docs) >= 12:
                break

        context = "\n\n".join(
            [
                f"Name: {doc.metadata.get('name', '')}\n"
                f"Target Muscle: {doc.metadata.get('target_muscle', '')}\n"
                f"Description: {doc.metadata.get('description', '')}\n"
                f"Difficulty: {doc.metadata.get('difficulty', '')}\n"
                f"Type: {doc.metadata.get('type', '')}\n"
                f"Image: {doc.metadata.get('image_url', '')}\n"
                f"Video: {doc.metadata.get('video_url', '')}"
                for doc in unique_docs
            ]
        ) if unique_docs else "No suitable exercises found."

        try:
            result = self.llm_chain.invoke({
                "context": context,
                "query": query,
                "age": age,
                "gender": gender,
                "fitness_goal": fitness_goal,
                "experience_level": experience_level,
                "available_equipment": available_equipment,
                "health_conditions": health_conditions
            })
            logging.info(f"LLM output: {result}")
            return {"plan": result or [], "source_documents": unique_docs}
        except Exception as e:
            logging.error(f"Error in LLM invocation: {str(e)}")
            return {"plan": [], "source_documents": unique_docs}

def get_fitness_plan_chain() -> FitnessPlanChain:
    # Locate exercise.json relative to current file
    base_dir = os.path.dirname(__file__)
    exercise_file = os.path.join(base_dir, "exercise.json")

    if not os.path.isfile(exercise_file):
        raise FileNotFoundError(f"Missing {exercise_file}")

    with open(exercise_file, "r", encoding="utf-8") as f:
        exercises = json.load(f)

    # Prepare the text chunks and metadata
    texts = [
        (
            f"Name: {ex['name']}\n"
            f"Target Muscle: {ex['target_muscle']}\n"
            f"Description: {ex['description']}\n"
            f"Difficulty: {ex['difficulty']}\n"
            f"Type: {ex['type']}\n"
            f"Image: {ex['image_url']}\n"
            f"Video: {ex['video_url']}"
        )
        for ex in exercises
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_query",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    persist_dir = os.path.join(base_dir, "chroma_fitness")

    if os.path.exists(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        vector_store = Chroma.from_texts(
            texts=texts,
            # Removed embedding_function=embeddings to avoid conflict
            metadatas=exercises,
            ids=[ex["name"] for ex in exercises],
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
            embedding=embeddings  # Pass embeddings as 'embedding' instead
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 12})

    # Prompt setup
    prompt = PromptTemplate(
        input_variables=[
            "context", "query", "age", "gender", "fitness_goal", "experience_level", "available_equipment", "health_conditions"
        ],
        template="""
You are a certified fitness coach. Create a 7-day JSON workout plan but keep rest days in between for those days return empty json based on the provided exercise data and user input. Return valid JSON only.

Exercise data:
{context}

User details:
- Age: {age}
- Gender: {gender}
- Fitness Goal: {fitness_goal}
- Experience Level: {experience_level}
- Available Equipment: {available_equipment}
- Health Conditions: {health_conditions}

Create a 7-day plan with 4–5 exercises each day. Each exercise must include:
- name, target_muscle, description, difficulty, type, image_url, video_url, reps

Return JSON like this:
[
  {{
    "day": "Day 1",
    "exercises": [
      {{
        "name": "", "target_muscle": "", "description": "", "difficulty": "", 
        "type": "", "image_url": "", "video_url": "", "reps": ""
      }}
    ]
  }},
  ...
  {{
    "day": "Day 7",
    "exercises": [ ... ]
  }}
]

If no valid exercises are found, return []
"""
    )

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        response_mime_type="application/json"
    )

    llm_chain = prompt | llm | JsonOutputParser()

    return FitnessPlanChain(retriever=retriever, llm_chain=llm_chain)