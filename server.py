from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from recomendationTrainer import RecommenderTrainer
from dqnRecommender import DQNRecommender  # noqa: F401  # used by trainer
from drlDataLoader import DRLDataAdapter
from dummyDataCreator import RecommenderDataGenerator

import numpy as np


app = FastAPI(title="DRL Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InitRequest(BaseModel):
    n_users: int = 500
    n_items: int = 100
    n_interactions: int = 10000
    save_output: bool = False
    output_dir: str = "datasets/generated"


class TrainRequest(BaseModel):
    episodes: int = 50


class RecommendRequest(BaseModel):
    user_id: int
    n_recommendations: int = 5


# Global trainer instance (simple demo scope)
trainer: Optional[RecommenderTrainer] = None
pipeline_info: Optional[Dict[str, Any]] = None
epsilon_history: List[float] = []


def build_trainer(n_users: int, n_items: int, n_interactions: int, save_output: bool = False, output_dir: str = "datasets/generated") -> RecommenderTrainer:
    generator = RecommenderDataGenerator(n_users=n_users, n_items=n_items)
    # Generate data without plotting or saving for speed
    generator.generate_users()
    generator.generate_items()
    generator.generate_interactions(n_interactions=n_interactions)
    generator.create_user_item_matrix()
    generator.generate_content_features()
    drl_data = generator.get_data_for_drl()
    if save_output:
        generator.save_data("recommender_data", output_dir=output_dir)

    # Create base trainer and swap in enhanced environment
    new_trainer = RecommenderTrainer(n_users=n_users, n_items=n_items)
    adapter = DRLDataAdapter(drl_data)
    enhanced_env = adapter.create_enhanced_environment()
    new_trainer.set_environment(enhanced_env)
    # Build pipeline info snapshot
    global pipeline_info, epsilon_history
    epsilon_history = []
    total_pairs = n_users * n_items
    num_interactions = int(len(generator.interactions_df))
    sparsity_percent = float((1 - num_interactions / total_pairs) * 100.0) if total_pairs > 0 else 0.0
    rating_count = int(generator.interactions_df['rating'].notna().sum())
    pipeline_info = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "num_interactions": num_interactions,
        "sparsity_percent": sparsity_percent,
        "rating_count": rating_count,
        "interaction_matrix_shape": list(generator.interaction_matrix.shape) if hasattr(generator, 'interaction_matrix') else None,
        "rating_matrix_shape": list(generator.rating_matrix.shape) if hasattr(generator, 'rating_matrix') else None,
        "item_feature_shape": list(generator.item_content_features.shape) if hasattr(generator, 'item_content_features') else None,
        "user_feature_shape": list(generator.user_content_features.shape) if hasattr(generator, 'user_content_features') else None,
        "state_dim": int(new_trainer.env.get_state_dim()),
    }
    return new_trainer


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/init")
def api_init(req: InitRequest) -> Dict[str, Any]:
    global trainer
    trainer = build_trainer(req.n_users, req.n_items, req.n_interactions, save_output=req.save_output, output_dir=req.output_dir)
    return {
        "message": "initialized",
        "n_users": req.n_users,
        "n_items": req.n_items,
        "state_dim": trainer.env.get_state_dim(),
        "saved": req.save_output,
        "output_dir": req.output_dir if req.save_output else None,
    }


@app.post("/api/train")
def api_train(req: TrainRequest) -> Dict[str, Any]:
    global trainer, epsilon_history
    if trainer is None:
        # Lazy init with defaults
        trainer = build_trainer(500, 100, 10000)

    progress: List[Dict[str, Any]] = []

    def on_episode_end(episode: int, total_reward: float, steps: int, epsilon: float):
        progress.append(
            {
                "episode": episode,
                "total_reward": float(total_reward),
                "steps": int(steps),
                "epsilon": float(epsilon),
            }
        )
        epsilon_history.append(float(epsilon))

    trainer.train(episodes=req.episodes, on_episode_end=on_episode_end)

    # Compute moving average of rewards (last 100 across all history)
    rewards = trainer.episode_rewards
    window = min(100, len(rewards))
    moving_avg = []
    if window > 0:
        cumsum = np.cumsum(np.insert(np.array(rewards, dtype=float), 0, 0.0))
        for i in range(window - 1, len(rewards)):
            avg = (cumsum[i + 1] - cumsum[i + 1 - window]) / window
            moving_avg.append(float(avg))

    return {
        "message": "training_complete",
        "episodes_run": req.episodes,
        "progress": progress,
        "metrics": {
            "episode_rewards": [float(r) for r in trainer.episode_rewards],
            "episode_lengths": [int(l) for l in trainer.episode_lengths],
            "moving_avg_rewards": moving_avg,
            "epsilon_history": epsilon_history,
        },
    }


@app.get("/api/metrics")
def api_metrics() -> Dict[str, Any]:
    global trainer, epsilon_history
    if trainer is None:
        raise HTTPException(status_code=400, detail="Trainer not initialized. Call /api/init first.")
    return {
        "episode_rewards": [float(r) for r in trainer.episode_rewards],
        "episode_lengths": [int(l) for l in trainer.episode_lengths],
        "epsilon_history": epsilon_history,
    }


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest) -> Dict[str, Any]:
    global trainer
    if trainer is None:
        raise HTTPException(status_code=400, detail="Trainer not initialized. Call /api/init first.")
    recs = trainer.recommend_for_user(req.user_id, n_recommendations=req.n_recommendations)
    return {"user_id": req.user_id, "recommendations": [int(x) for x in recs]}


@app.get("/api/pipeline")
def api_pipeline() -> Dict[str, Any]:
    global trainer, pipeline_info
    if trainer is None or pipeline_info is None:
        raise HTTPException(status_code=400, detail="Trainer not initialized. Call /api/init first.")
    return pipeline_info

