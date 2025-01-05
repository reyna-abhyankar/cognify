import uuid

class TrialLog:
    def __init__(
        self,
        params: dict[str, any],
        id: str = None,
        score: float = 0.0,
        price: float = 0.0,
        eval_cost: float = 0.0,
        finished: bool = False,
    ):
        self.id: str = id or uuid.uuid4().hex
        self.params = params
        self.score = score
        self.price = price
        self.eval_cost = eval_cost
        self.finished = finished

    def to_dict(self):
        return {
            "id": self.id,
            "bo_trial_id": self.bo_trial_id,
            "params": self.params,
            "score": self.score,
            "price": self.price,
            "eval_cost": self.eval_cost,
            "finished": self.finished,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    