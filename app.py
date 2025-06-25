import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from modules.workout_chain import get_fitness_plan_chain

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# FastAPI setup
app = FastAPI(title="HealthyEats RAG Fitness API")

# Initialize the fitness plan chain
fitness_chain = get_fitness_plan_chain()

# Pydantic models for request and response
class PlanRequest(BaseModel):
    age: int
    gender: str
    fitness_goal: str
    experience_level: str
    available_equipment: str
    health_conditions: str

class PlanResponse(BaseModel):
    plan: list  # JSON structure for 7-day plan

@app.post("/plan", response_model=PlanResponse)
async def create_plan(req: PlanRequest):
    try:
        inputs = {
            "age": req.age,
            "gender": req.gender,
            "fitness_goal": req.fitness_goal,
            "experience_level": req.experience_level,
            "available_equipment": req.available_equipment,
            "health_conditions": req.health_conditions
        }
        result = fitness_chain.invoke(inputs)
        logging.info(f"Chain output: {result}")
        return PlanResponse(plan=result["plan"])
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)