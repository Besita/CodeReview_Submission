from http import server
import uvicorn
from fastapi import FastAPI
from openenv.core.env_server import create_app, create_fastapi_app
from server.environment import CodeReviewEnv
from models import CodeReviewAction, CodeReviewObservation

# 1. Manually create the instance once
create_env = CodeReviewEnv(max_steps=5) 

# 2. Wrap it in a function that ALWAYS returns that same instance
def get_env():
    return create_env

'''#method1
def create_env():
    # Pass your max_steps here
    return CodeReviewEnv(max_steps=5)'''


# Create your environment
app = create_fastapi_app(
    get_env,   # ✅ pass instance
    action_cls=CodeReviewAction,
    observation_cls=CodeReviewObservation,
)

# Add it here!
@app.get("/")
async def root():
    return {"message": "Server is running!"}

import subprocess

@app.on_event("startup")
def run_inference():
    try:
        subprocess.run(["python", "inference.py"], check=True)
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860,workers=1)


   
    '''from env_server import create_app
from server.environment import CodeReviewEnv
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState

# Required entrypoint for openenv.yaml
#app = create_app(CodeReviewEnv)
app = create_app(
    CodeReviewEnv,          # environment class
    CodeReviewAction,       # action model class
    CodeReviewObservation   # observation model class
)

def main():
    app.run(host="0.0.0.0", port=7860)

# Required for OpenEnv validation
if __name__ == "__main__":
    main()'''

'''
from openenv.core.env_server import create_app
from server.environment import CodeReviewEnv
from models import CodeReviewAction, CodeReviewObservation

# Create the OpenEnv FastAPI app
app = create_app(
    CodeReviewEnv,          # Environment class
    CodeReviewAction,       # Action model class
    CodeReviewObservation   # Observation model class
)

# Required for standalone execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)'''